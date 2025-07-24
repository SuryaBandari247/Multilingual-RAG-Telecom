"""
RAG System Evaluation using BERTScore

This python script evaluates the performance of different LLM models in the multilingual RAG system
using BERTScore metrics (precision, recall, F1) to measure semantic similarity between
generated answers and reference text across multiple languages.

The evaluation covers 5 LLM models across 4 languages (German, French, Italian, English)
with 100 samples per language per model (400 total samples per model).
"""

import pandas as pd
import os
from evaluate import load
from typing import Dict, List, Tuple, Any


def load_bertscore_metric():
    """
    Load BERTScore metric from Hugging Face evaluate library.
    
    Returns:
        evaluate.Metric: BERTScore metric object for semantic similarity evaluation
    """
    return load("bertscore")


def get_language_mappings():
    """
    Get language code mappings for BERTScore evaluation.
    
    Returns:
        Two dictionaries containing:
            - ISO language codes (de, fr, it, en) for BERTScore
            - Full language names for output formatting
    """
    lang_map = {0: "de", 1: "fr", 2: "it", 3: "en"}
    lang_full = {0: "German", 1: "French", 2: "Italian", 3: "English"}
    return lang_map, lang_full


def extract_model_name(filename):
    """
    Extract model name from input filename.
    
    Args:
        filename (str): Input CSV filename containing model evaluation results
    
    Returns:
        str: Clean model name extracted from filename
    """
    return filename.replace("df_ragas_", "").replace("_with_cleaned_text.csv", "")


def evaluate_model_language(df, bertscore, lang_idx, 
                          lang_map, lang_full, 
                          model_name):
    """
    Evaluate BERTScore metrics for a specific model-language combination.
    
    Args:
        df: DataFrame containing model evaluation data
        bertscore: BERTScore metric object
        lang_idx: Language index (0-3 for German, French, Italian, English)
        lang_map: ISO language codes mapping
        lang_full: Full language names mapping
        model_name: Name of the model being evaluated
    
    Returns:
        Dictionary containing evaluation results with keys:
            - model: Model name
            - language: Full language name
            - bertscore_precision: Average precision score
            - bertscore_recall: Average recall score
            - bertscore_f1: Average F1 score
    """
    start = lang_idx * 100
    end = (lang_idx + 1) * 100
    lang_code = lang_map[lang_idx]
    lang_name = lang_full[lang_idx]
    
    preds = df['answer'].iloc[start:end].astype(str).tolist()
    refs = df['cleaned_text'].iloc[start:end].astype(str).tolist()
    
    bertscore_result = bertscore.compute(predictions=preds, references=refs, lang=lang_code)
    
    avg_precision = sum(bertscore_result['precision']) / len(bertscore_result['precision'])
    avg_recall = sum(bertscore_result['recall']) / len(bertscore_result['recall'])
    avg_f1 = sum(bertscore_result['f1']) / len(bertscore_result['f1'])
    
    print(f"{model_name} - {lang_name}: F1={avg_f1:.4f}")
    
    return {
        "model": model_name,
        "language": lang_name,
        "bertscore_precision": avg_precision,
        "bertscore_recall": avg_recall,
        "bertscore_f1": avg_f1
    }


def process_model_file(file_path, filename, bertscore, 
                      lang_map, lang_full):
    """
    Process a single model's evaluation file and compute BERTScore metrics.
    
    Args:
        file_path: Full path to the model evaluation CSV file
        filename: Name of the file being processed
        bertscore: BERTScore metric object
        lang_map: ISO language codes mapping
        lang_full: Full language names mapping
    
    Returns:
        List of evaluation results for all languages in the model
    
    Raises:
        AssertionError: If the file doesn't contain exactly 400 rows
    """
    model_name = extract_model_name(filename)
    print(f"Processing {file_path}...")
    
    df = pd.read_csv(file_path)
    assert len(df) == 400, f"Expected 400 rows in {filename}, got {len(df)}"
    
    model_results = []
    for i in range(4):
        result = evaluate_model_language(df, bertscore, i, lang_map, lang_full, model_name)
        model_results.append(result)
    
    return model_results


def main():
    """
    Main function to evaluate all models using BERTScore metrics.
    
    This function orchestrates the complete BERTScore evaluation pipeline:
    1. Loads BERTScore metric and language mappings
    2. Processes each model's evaluation file (5 models total)
    3. Computes semantic similarity metrics for each language
    4. Aggregates results and saves to CSV
    
   
    Output:
        - rag-evaluation-bertscore.csv: Comprehensive evaluation results
    """
    input_dir = r'../../dataset/generated'
    output_path = r'results/rag-evaluation-bertscore.csv'
    
    input_files = [
        "df_ragas_granite3.3_8b_with_cleaned_text.csv",
        "df_ragas_llama3_8b_with_cleaned_text.csv",
        "df_ragas_mistral_7b_with_cleaned_text.csv",
        "df_ragas_zephyr_7b_with_cleaned_text.csv",
        "df_ragas_wizardlm2_7b_with_cleaned_text.csv"
    ]
    
    bertscore = load_bertscore_metric()
    lang_map, lang_full = get_language_mappings()
    
    results = []
    for filename in input_files:
        file_path = os.path.join(input_dir, filename)
        model_results = process_model_file(file_path, filename, bertscore, lang_map, lang_full)
        results.extend(model_results)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\nBERTScore results saved to {output_path}")


if __name__ == "__main__":
    main()

