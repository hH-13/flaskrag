from ..base import EmbeddingGenerator


class MockEmbedding(EmbeddingGenerator):
    def generate_embedding(self, data: str, **kwargs) -> list[float]:
        return [1.0, 2.0, 3.0, 4.0, 5.0]
