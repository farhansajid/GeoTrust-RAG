"""
Trustworthiness & Calibrated Selective Prediction Engine.
Implements:
1. Multi-Sample Semantic Clustering & Semantic Entropy (SE)
2. Evidence-Conformity Scoring (CS)
3. Combined Calibrated Epistemic Confidence
4. Reject-Option (Abstention) Rule
5. Risk-Coverage and Expected Calibration Error (ECE) Analytics
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class TrustworthinessEngine:
    def __init__(self, encoder, alpha=0.6, default_threshold=0.65):
        """
        encoder: SentenceTransformer instance (CPU)
        alpha: Weight balancing Semantic Entropy vs Evidence Conformity
        default_threshold: Calibrated decision threshold tau
        """
        self.encoder = encoder
        self.alpha = alpha
        self.default_threshold = default_threshold

    def compute_semantic_clusters(self, candidate_answers, similarity_threshold=0.72):
        """
        Groups M candidate answers into semantic equivalence classes based on:
        1. Exact numerical / categorical token equivalence (for VQA answers)
        2. Cosine embedding similarity
        """
        if not candidate_answers:
            return []
        if len(candidate_answers) == 1:
            return [[candidate_answers[0]]]

        import re
        def extract_core_entity(text):
            # Extract numbers if present
            nums = re.findall(r'\b\d+\b', text)
            if nums:
                return nums[-1]  # Return the primary numerical answer
            # Extract key categorical labels
            text_lower = text.lower()
            for key_term in ["rural", "urban", "yes", "no", "level 1", "level 2", "level 3", "level 4", "flooded", "non-flooded", "overflowing"]:
                if key_term in text_lower:
                    return key_term
            return None

        core_entities = [extract_core_entity(ans) for ans in candidate_answers]

        embeddings = self.encoder.encode(candidate_answers, convert_to_numpy=True, normalize_embeddings=True)
        sim_matrix = cosine_similarity(embeddings)

        clusters = []
        assigned = set()

        for i in range(len(candidate_answers)):
            if i in assigned:
                continue
            cluster = [candidate_answers[i]]
            assigned.add(i)
            for j in range(i + 1, len(candidate_answers)):
                if j in assigned:
                    continue
                # Match if same core entity (e.g. both '5' or both 'rural') OR embedding similarity >= threshold
                same_core = (core_entities[i] is not None and core_entities[i] == core_entities[j])
                high_sim = sim_matrix[i, j] >= similarity_threshold
                if same_core or high_sim:
                    cluster.append(candidate_answers[j])
                    assigned.add(j)
            clusters.append(cluster)

        return clusters

    def compute_semantic_entropy(self, clusters, num_total_samples):
        r"""
        Calculates Semantic Entropy: SE(x) = - \sum p(c_k) log p(c_k)
        """
        if num_total_samples <= 1 or len(clusters) <= 1:
            return 0.0

        probs = [len(c) / num_total_samples for c in clusters]
        se = -sum(p * np.log(p + 1e-12) for p in probs)
        max_entropy = np.log(num_total_samples)
        normalized_se = float(np.clip(se / max_entropy, 0.0, 1.0))
        return normalized_se

    def compute_evidence_conformity(self, answer, retrieved_passages):
        """
        Computes maximum semantic alignment between the generated answer and retrieved evidence passages.
        """
        if not retrieved_passages:
            return 0.5  # Neutral default if no passages provided

        import re
        citations = re.findall(r'\[(Doc-\d+)\]', answer)
        retrieved_ids = [p['doc_id'] for p in retrieved_passages]
        has_matching_citation = any(cit in retrieved_ids for cit in citations)

        ans_emb = self.encoder.encode([answer], convert_to_numpy=True, normalize_embeddings=True)
        doc_texts = [f"{p['topic']}: {p['content']}" for p in retrieved_passages]
        doc_embs = self.encoder.encode(doc_texts, convert_to_numpy=True, normalize_embeddings=True)

        sims = cosine_similarity(ans_emb, doc_embs)[0]
        max_sim = float(np.max(sims))
        
        # If the answer explicitly cites a retrieved document and aligns with it, scale conformity
        if has_matching_citation:
            conformity = float(np.clip(max_sim + 0.35, 0.0, 1.0))
        else:
            conformity = float(np.clip(max_sim, 0.0, 1.0))
            
        return conformity

    def evaluate_confidence(self, candidate_answers, retrieved_passages):
        """
        Evaluates unified epistemic confidence C(x) = (1 - SE_norm)^alpha * CS^(1 - alpha)
        """
        num_samples = len(candidate_answers)
        clusters = self.compute_semantic_clusters(candidate_answers)
        se_norm = self.compute_semantic_entropy(clusters, num_samples)

        # Select primary representative answer (majority cluster)
        largest_cluster = max(clusters, key=len)
        primary_answer = largest_cluster[0]

        cs_score = self.compute_evidence_conformity(primary_answer, retrieved_passages)

        # Confidence metric
        confidence = float(((1.0 - se_norm) ** self.alpha) * (cs_score ** (1.0 - self.alpha)))
        confidence = float(np.clip(confidence, 0.0, 1.0))

        return {
            "primary_answer": primary_answer,
            "confidence": confidence,
            "semantic_entropy_norm": se_norm,
            "evidence_conformity": cs_score,
            "cluster_count": len(clusters)
        }

    def selective_predict(self, candidate_answers, retrieved_passages, threshold=None):
        """
        Applies the calibrated reject-option rule:
        If confidence < tau -> ABSTAIN
        """
        tau = threshold if threshold is not None else self.default_threshold
        eval_result = self.evaluate_confidence(candidate_answers, retrieved_passages)

        primary_answer = eval_result["primary_answer"]
        confidence = eval_result["confidence"]

        # Check explicit abstention signals in generation
        lower_ans = primary_answer.lower()
        has_abstain_intent = any(phrase in lower_ans for phrase in [
            "abstain", "insufficient evidence", "cannot be determined", "obscured", "unverified"
        ])

        if confidence < tau or has_abstain_intent:
            return {
                "decision": "ABSTAIN",
                "output_text": "ABSTAIN: Insufficient visual or retrieved evidence to ensure reliable decision.",
                "confidence": confidence,
                "is_abstained": True,
                "metadata": eval_result
            }
        else:
            return {
                "decision": "ANSWER",
                "output_text": primary_answer,
                "confidence": confidence,
                "is_abstained": False,
                "metadata": eval_result
            }
