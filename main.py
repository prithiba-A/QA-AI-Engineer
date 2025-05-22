import argparse
import json
import os
from crawler import crawl_and_extract
from embedder import Embedder
from qa_agent import generate_answer
from functools import lru_cache
import numpy as np

# In-memory cache for previously asked questions
cache = {}

# Optional: use lru_cache for function-level caching
@lru_cache(maxsize=128)
def cached_answer(question, context):
    return generate_answer(question, context)

def get_confidence_score(query_vec, result_vec):
    from numpy.linalg import norm
    score = 1 - norm(query_vec - result_vec) / 2  # Normalize 0-1
    return round(score * 100, 2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', nargs='+', help='One or more documentation URLs', required=True)
    args = parser.parse_args()

    all_documents = {}
    print("\nStarting crawl...")
    for url in args.url:
        print(f" Crawling: {url}")
        docs = crawl_and_extract(url, max_pages=5)
        all_documents.update(docs)

    print(f"\nTotal pages crawled: {len(all_documents)}")

    print("\nBuilding index...")
    embedder = Embedder()
    embedder.build_index(all_documents)

    print("\nReady to answer your questions! Type 'exit' or 'quit' to stop.")
    while True:
        question = input("\nQ: ")
        if question.lower() in ['exit', 'quit']:
            break

        if question in cache:
            print("(cached)")
            print(f"\nA: {cache[question]['answer']}\nConfidence: {cache[question]['confidence']}%\nSource: {cache[question]['source']}\n")
            continue

        context_results = embedder.search(question)
        combined_context = "\n".join([chunk for chunk, _ in context_results])
        answer = cached_answer(question, combined_context)

        # Encode query and calculate confidence using stored vectors directly
        q_vector = np.array([embedder.model.encode(question)]).astype('float32')
        D, I = embedder.index.search(q_vector, 1)
        top_vector = embedder.vectors[I[0][0]]  # get vector directly from stored vectors
        confidence = get_confidence_score(q_vector[0], top_vector)

        source_url = context_results[0][1] if context_results else "Unknown"
        
        print(f"\nA: {answer}\nConfidence: {confidence}%\nSource: {source_url}\n")

        cache[question] = {"answer": answer, "confidence": confidence, "source": source_url}

if __name__ == "__main__":
    main()
