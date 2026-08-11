from rag_retriever import get_relevant_context

query = "What are the recommended approaches for managing stress?"

context = get_relevant_context(query, k=3)

print("\n--- RAG CONTEXT ---")
print(context)