"""
RAG Retriever Evaluation using Semantic Context Precision

This python script evaluates retriever performance by measuring semantic similarity
between retrieved contexts and reference answers using multilingual sentence transformers.
"""

import pandas as pd
import ast
from sentence_transformers import SentenceTransformer, util

# Load multilingual sentence transformer model for semantic similarity
model = SentenceTransformer('sentence-transformers/static-similarity-mrl-multilingual-v1')

def parse_contexts(contexts_str):
    """
    Parse and extract page content from serialized context strings.
    
    Args:
        contexts_str: Serialized string containing LangChain document objects
    
    Returns:
        list: Extracted page content strings
    """
    try:
        contexts = ast.literal_eval(contexts_str)
        if isinstance(contexts, list):
            out = []
            for c in contexts:
                if isinstance(c, str) and "page_content=" in c:
                    start = c.find('page_content="')
                    if start != -1:
                        start += len('page_content="')
                        end = c.find('")', start)
                        if end != -1:
                            out.append(c[start:end])
                        else:
                            out.append(c[start:])
                    else:
                        out.append(c)
                else:
                    out.append(str(c))
            return out
        else:
            return [str(contexts)]
    except Exception:
        return [str(contexts_str)]

def semantic_precision(retrieved_docs, answer_text, threshold=0.7):
    """
    Calculate semantic precision of retrieved documents against reference answer.
    
    Args:
        retrieved_docs: List of retrieved document texts
        answer_text: Reference answer text
        threshold: Similarity threshold for relevance (default 0.7)
    
    Returns:
        float: Precision score (0.0 to 1.0)
    """
    if not retrieved_docs:
        return 0.0
    answer_emb = model.encode(answer_text, convert_to_tensor=True)
    relevant = 0
    for doc in retrieved_docs:
        doc_emb = model.encode(doc, convert_to_tensor=True)
        sim = util.cos_sim(answer_emb, doc_emb).item()
        if sim > threshold:
            relevant += 1
    return relevant / len(retrieved_docs)

def main():
    """
    Main function to evaluate retriever semantic context precision across languages.

    """
    df = pd.read_csv(
        r"../../dataset/generated/df_ragas_llama3_8b_with_cleaned_text.csv"
    )

    # Assign language based on row index
    languages = ['German'] * 100 + ['French'] * 100 + ['Italian'] * 100 + ['English'] * 100
    df['language'] = languages

    precisions = []
    for idx, row in df.iterrows():
        retrieved_docs = parse_contexts(row["contexts"])
        answer_text = row["cleaned_text"]
        precision = semantic_precision(retrieved_docs, answer_text)
        precisions.append(precision)
        print(f"[{idx+1}/{len(df)}] {row['language']} - Semantic context precision: {precision:.2f}")

    df["semantic_context_precision"] = precisions

    # Compute and print average per language
    for lang in ['German', 'French', 'Italian', 'English']:
        lang_precisions = df[df['language'] == lang]["semantic_context_precision"]
        avg_precision = lang_precisions.mean() if not lang_precisions.empty else 0
        print(f"\nAverage semantic context precision for {lang}: {avg_precision:.3f}")

    df.to_csv("semantic_context_precision_results_by_language.csv", index=False)
    print("Results saved to semantic_context_precision_results_by_language.csv")

if __name__ == "__main__":
    main()