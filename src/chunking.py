from __future__ import annotations
import math
import re

class FixedSizeChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks

class SentenceChunker:
    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text: return []
        # Split on sentence boundaries using regex
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk = " ".join(sentences[i : i + self.max_sentences_per_chunk])
            if chunk:
                chunks.append(chunk.strip())
        return chunks

class RecursiveChunker:
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        return self._split(text, self.separators)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size or not separators:
            return [text]

        sep = separators[0]
        remaining = separators[1:]
        
        # Split text by the current separator
        parts = text.split(sep) if sep != "" else list(text)
        final_chunks = []
        current_buffer = ""

        for part in parts:
            # If adding part exceeds size, process the buffer
            if len(current_buffer) + len(sep) + len(part) > self.chunk_size and current_buffer:
                final_chunks.extend(self._split(current_buffer, remaining))
                current_buffer = ""
            
            if current_buffer:
                current_buffer += sep + part
            else:
                current_buffer = part

        if current_buffer:
            final_chunks.extend(self._split(current_buffer, remaining))
            
        return final_chunks

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    norm_a = math.sqrt(sum(x*x for x in vec_a))
    norm_b = math.sqrt(sum(x*x for x in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return _dot(vec_a, vec_b) / (norm_a * norm_b)

class ChunkingStrategyComparator:
    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size)
        }
        results = {}
        for name, Chunker in strategies.items():
            chunks = Chunker.chunk(text)
            avg_len = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
            results[name] = {"count": len(chunks), "avg_length": avg_len, "chunks": chunks}
        return results