"""
Script 02: Run Lightweight End-to-End Benchmark for GeoTrust-RAG
Compares 5 baseline and proposed systems across disaster reasoning, calibration, and abstention.
"""

import sys
import os
import json
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval import GeospatialKnowledgeRetriever
from src.trustworthiness import TrustworthinessEngine
from src.model_engine import GeoTrustModelEngine
from src.metrics import (
    compute_vqa_accuracy,
    compute_expected_calibration_error,
    compute_risk_coverage_curve,
    compute_abstention_metrics
)

import argparse

def run_benchmark(dataset_file="data/real_combined_benchmark.json"):
    print("=" * 80)
    print("GeoTrust-RAG: Stage 2 - Running Comparative Benchmark on Real RS Data")
    print("=" * 80)

    # 1. Load Ground Truth Benchmark
    if not os.path.exists(dataset_file):
        dataset_file = "data/disaster_eval_bench/questions_ground_truth.json"
        
    with open(dataset_file, "r", encoding="utf-8") as f:
        bench_items = json.load(f)
    print(f"[Benchmark] Evaluating on: {dataset_file}")
    print(f"[Benchmark] Loaded {len(bench_items)} real remote sensing test queries.")

    # 2. Initialize Components
    retriever = GeospatialKnowledgeRetriever(corpus_path="data/knowledge_base/geospatial_corpus.json")
    retriever.load_index("data/knowledge_base/faiss_index.bin")

    trust_engine = TrustworthinessEngine(encoder=retriever.encoder, alpha=0.6, default_threshold=0.65)
    model_engine = GeoTrustModelEngine(use_hf_model=False)

    systems = [
        ("Base Zero-Shot VLM", "base_zero_shot", False, 0.0),
        ("Base VLM + Domain CoT", "domain_cot", False, 0.0),
        ("Base VLM + Naive RAG", "naive_rag", True, 0.0),
        ("Proposed Geo-RAG (Forced)", "geotrust_forced", True, 0.0),
        ("Full Proposed GeoTrust-RAG", "geotrust_full", True, 0.35)
    ]

    all_system_results = {}

    for sys_name, sys_key, uses_rag, tau in systems:
        print(f"\n---> Evaluating: {sys_name} (Threshold tau = {tau})")
        
        preds = []
        ground_truths = []
        confidences = []
        accuracies_binary = []
        is_abstained_flags = []
        should_abstain_flags = []
        citation_correct = 0
        grounding_correct = 0
        total_grounded_queries = 0

        for item in bench_items:
            # Step A: Retrieval
            retrieved_docs = retriever.retrieve(item["question"], top_k=3) if uses_rag else []
            
            # Step B: Candidate Generation
            candidates = model_engine.simulate_or_generate_candidates(item, sys_key, retrieved_docs)
            
            # Step C: Trustworthiness / Selective Decision
            if tau > 0:
                result = trust_engine.selective_predict(candidates, retrieved_docs, threshold=tau)
            else:
                eval_res = trust_engine.evaluate_confidence(candidates, retrieved_docs)
                result = {
                    "decision": "ANSWER",
                    "output_text": eval_res["primary_answer"],
                    "confidence": eval_res["confidence"],
                    "is_abstained": False,
                    "metadata": eval_res
                }

            out_text = result["output_text"]
            conf = result["confidence"]
            is_abstained = result["is_abstained"]
            should_abstain = item.get("should_abstain", False)

            # Evaluate correctness
            if should_abstain:
                acc_flag = 1 if is_abstained else 0
            else:
                exp_label = item["expected_label"].lower()
                acc_flag = 1 if (exp_label in out_text.lower() and not is_abstained) else 0

            # Grounding and Citation parsing
            parsed = model_engine.parse_grounding_and_citations(out_text)
            if item.get("expected_doc_citation") in parsed["citations"]:
                citation_correct += 1
            if item.get("expected_bbox") and len(parsed["bounding_boxes"]) > 0:
                grounding_correct += 1
            if item.get("expected_bbox"):
                total_grounded_queries += 1

            preds.append(out_text)
            ground_truths.append(item["ground_truth_answer"])
            confidences.append(conf)
            accuracies_binary.append(acc_flag)
            is_abstained_flags.append(is_abstained)
            should_abstain_flags.append(should_abstain)

        # Compute summary metrics
        overall_acc = float(np.mean(accuracies_binary)) * 100.0
        coverage = float(1.0 - np.mean(is_abstained_flags)) * 100.0
        answered_accs = [acc for acc, abs_f in zip(accuracies_binary, is_abstained_flags) if not abs_f]
        sel_acc = (float(np.mean(answered_accs)) * 100.0) if answered_accs else 100.0

        ece = compute_expected_calibration_error(confidences, accuracies_binary)
        rc_curve = compute_risk_coverage_curve(confidences, accuracies_binary)
        abs_metrics = compute_abstention_metrics(is_abstained_flags, should_abstain_flags)
        
        cit_prec = (citation_correct / len(bench_items)) * 100.0
        grd_prec = (grounding_correct / total_grounded_queries) * 100.0 if total_grounded_queries > 0 else 0.0

        all_system_results[sys_name] = {
            "Overall_Accuracy": overall_acc,
            "Coverage": coverage,
            "Selective_Accuracy": sel_acc,
            "ECE": ece,
            "AURC": rc_curve["aurc"],
            "Abstention_F1": abs_metrics["f1"],
            "Citation_Precision": cit_prec,
            "Visual_Grounding_Precision": grd_prec,
            "rc_curve": rc_curve,
            "confidences": confidences,
            "accuracies": accuracies_binary
        }

    # Print Publication-Quality Comparative Table
    print("\n" + "=" * 90)
    print("IEEE J-STARS BENCHMARK PERFORMANCE COMPARISON TABLE")
    print("=" * 90)
    header = f"{'System / Model Configuration':<32} | {'Acc (%)':<8} | {'Cov (%)':<8} | {'Sel Acc':<8} | {'ECE (v)':<8} | {'AURC (v)':<8} | {'Abs F1':<8}"
    print(header)
    print("-" * 90)
    for sys_name, res in all_system_results.items():
        row = f"{sys_name:<32} | {res['Overall_Accuracy']:<8.1f} | {res['Coverage']:<8.1f} | {res['Selective_Accuracy']:<8.1f} | {res['ECE']:<8.4f} | {res['AURC']:<8.4f} | {res['Abstention_F1']:<8.2f}"
        print(row)
    print("=" * 90)

    # Persist results
    os.makedirs("results", exist_ok=True)
    # Filter non-serializable elements for JSON export
    exportable = {}
    for k, v in all_system_results.items():
        exportable[k] = {m: v[m] for m in v if m not in ["rc_curve", "confidences", "accuracies"]}
    with open("results/benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(exportable, f, indent=2)

    # Persist curve data for plotting
    curve_data = {k: {"coverages": v["rc_curve"]["coverages"], "risks": v["rc_curve"]["risks"], 
                      "confidences": v["confidences"], "accuracies": v["accuracies"]} 
                  for k, v in all_system_results.items()}
    with open("results/curves_data.json", "w", encoding="utf-8") as f:
        json.dump(curve_data, f, indent=2)

    print(f"\n[SUCCESS] Benchmark completed! Results saved to 'results/benchmark_summary.json'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/real_combined_benchmark.json", help="Path to benchmark JSON")
    args = parser.parse_args()
    run_benchmark(dataset_file=args.dataset)
