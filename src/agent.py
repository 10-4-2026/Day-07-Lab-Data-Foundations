from typing import Callable
from .store import EmbeddingStore

class KnowledgeBaseAgent:
    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # 1. Retrieve
        hits = self.store.search(question, top_k=top_k)
        context = "\n---\n".join([h["content"] for h in hits])
        
        # 2. Build Prompt
        prompt = (
            f"Context information is below.\n{context}\n"
            f"Given the context, please answer the question: {question}"
        )
        
        # 3. Generate
        return self.llm_fn(prompt)