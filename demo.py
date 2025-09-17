from flaskrag import (
    APIServer,
    BaseRAG,
    DatabaseClient,
    MockEmbedding,
    MockLLM,
    MockVectorDB,
)

RAG = BaseRAG(
    vdb=MockVectorDB(),
    embedding_generator=MockEmbedding(),
    llm=MockLLM(),
    db=DatabaseClient(url="duckdb:///:memory:"),
)

API = APIServer(rag=RAG, debug=True)

API.run()
