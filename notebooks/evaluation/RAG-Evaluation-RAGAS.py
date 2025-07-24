"""
RAG System Evaluation using RAGAS Framework

This python script evaluates the performance of multilingual RAG systems using the RAGAS 
(Retrieval Augmented Generation Assessment) framework. It processes evaluation 
data from multiple LLM models across different languages and computes key metrics
including faithfulness, answer relevancy, and context precision.

Dependencies:
    - pandas: Data manipulation and analysis
    - ragas: RAG evaluation framework
    - langchain_openai: OpenAI model integration
    - datasets: Hugging Face datasets for RAGAS compatibility
"""

import pandas as pd
import re
import time

from datasets import Dataset as HFDataset
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference

import ast

from langchain_openai import OpenAI
from ragas.llms import LangchainLLMWrapper
import os
import asyncio


def extract_page_contents(contexts_str):
    """
    Extract page_content strings from serialized LangChain document objects.
    
    Args:
        contexts_str (str): Serialized string containing document objects
    
    Returns:
        List[str]: Extracted page content strings, empty list if extraction fails
    """
    try:
        return re.findall(r'page_content="(.*?)"', contexts_str, re.DOTALL)
    except Exception as e:
        print(f"[ERROR] Failed to extract page_content: {e}")
        return []




def convert_result_to_dict(result):
    """
    Convert RAGAS EvaluationResult object to dictionary format.
    
    Args:
        result (Any): RAGAS EvaluationResult object
    
    Returns:
        Dict[str, Any]: Dictionary with faithfulness, answer_relevancy, context_precision scores
    """
    try:
        result_str = str(result)
        return ast.literal_eval(result_str)
    except Exception as e:
        print(f"[ERROR] Failed to convert EvaluationResult to dict: {e}")
        return {
            "faithfulness": getattr(result, "faithfulness", None),
            "answer_relevancy": getattr(result, "answer_relevancy", None)
        }


def evaluate_by_language(df, ragas_llm):
    """
    Perform RAGAS evaluation for each language in the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame with question, answer, contexts, language columns
        ragas_llm: LangChain-wrapped LLM model for evaluation
    
    Returns:
        pd.DataFrame: Results with language, faithfulness, answer_relevancy, context_precision
    """
    print("[INFO] Starting RAG evaluation...")
    languages = df["language"].unique()
    results = []

    # Define metric
    context_precision = LLMContextPrecisionWithoutReference(llm=ragas_llm)

    for lang in languages:
        print(f"[INFO] Evaluating language: {lang} with {len(df[df['language'] == lang])} records...")

        lang_df = df[df["language"] == lang].copy()
        rag_dataset = HFDataset.from_pandas(lang_df[["question", "answer", "contexts"]])

        try:
            result = evaluate(
                rag_dataset,
                metrics=[faithfulness, answer_relevancy, context_precision],
                llm=ragas_llm,
            )
            result_dict = convert_result_to_dict(result)
            faithfulness_score = result_dict.get("faithfulness", None)
            answer_relevancy_score = result_dict.get("answer_relevancy", None)
            context_precision_score = result_dict.get("context_precision", None)
            print(f"[INFO] Raw result for {lang}: {result_dict}")

        except Exception as e:
            print(f"[WARNING] Evaluation failed for {lang}: {e}")
            faithfulness_score = None
            answer_relevancy_score = None
            context_precision_score = None

        results.append({
            "language": lang,
            "faithfulness": faithfulness_score,
            "answer_relevancy": answer_relevancy_score,
            "context_precision": context_precision_score,
        })

    return pd.DataFrame(results)


def main():
    """
    Main function for multilingual RAG system evaluation using RAGAS framework.
    
    Processes 5 LLM models (Granite, Llama3, Mistral, Zephyr, WizardLM2) across 
    4 languages (German, French, Italian, English) with 100 samples per language.
    
    Evaluates faithfulness, answer relevancy, and context precision metrics.
    Saves results to individual CSV files per model.
    """
    start_time = time.time()
    
    llm = OpenAI(model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"),max_tokens=10000)
    ragas_llm = LangchainLLMWrapper(llm)
    
    input_dir = r"../../dataset/generated"

    input_files = [
        "df_ragas_granite3.3_8b.csv",
        "df_ragas_llama3_8b.csv",
        "df_ragas_mistral_7b.csv",
        "df_ragas_zephyr_7b.csv",
        "df_ragas_wizardlm2_7b.csv",
    ]

    for input_file in input_files:
        input_path = os.path.join(input_dir, input_file)
        print(f"\n[INFO] Processing file: {input_path}")
        model_name = input_file.replace("df_ragas_", "").replace(".csv", "")
        output_path = f"rag_evaluation_results_per_language_{model_name}.csv"

        df = pd.read_csv(input_path)
        print(f"[INFO] Loaded DataFrame with {len(df)} records.")

        print("[INFO] Extracting page content from 'contexts' column...")
        df['contexts'] = df['contexts'].apply(
            lambda x: extract_page_contents(x) if isinstance(x, str) else x
        )

        if not isinstance(df['contexts'].iloc[0], list):
            raise ValueError("[ERROR] Each value in 'contexts' must be a list of strings.")

        print("[INFO] Assigning language labels...")
        num_records = len(df)
        if num_records != 400:
            raise ValueError(f"[ERROR] Expected 400 records (100 per language), found {num_records}")
        df["language"] = (["German"] * 100) + (["French"] * 100) + (["Italian"] * 100) + (["English"] * 100)

        results_df = evaluate_by_language(df, ragas_llm)
        results_df.to_csv(output_path, index=False)
        print(f"[INFO] Results saved to {output_path}")

    elapsed_minutes = (time.time() - start_time) / 60
    print(f"[INFO] Total execution time: {elapsed_minutes:.2f} minutes")
    print("[SUCCESS] All evaluations completed.")

if __name__ == "__main__":
    main()