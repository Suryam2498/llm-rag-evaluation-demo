"""Small, dependency-light retriever: paragraph chunking plus TF-IDF similarity."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Chunk:
    source: str
    text: str


def load_chunks(kb_dir: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(kb_dir).glob("*.md")):
        for para in path.read_text(encoding="utf-8").split("\n\n"):
            para = para.strip()
            if para and not para.startswith("#"):
                chunks.append(Chunk(source=path.name, text=para))
    return chunks


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("Knowledge base is empty")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def search(self, query: str, k: int = 3) -> list[Chunk]:
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in ranked[:k] if scores[i] > 0]
