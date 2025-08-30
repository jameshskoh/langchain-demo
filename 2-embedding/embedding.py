from dotenv import load_dotenv
from qwen.embedding_client import QwenEmbeddings

if __name__ == "__main__":

    load_dotenv()

    embeddings = QwenEmbeddings(base_url="http://localhost:1234/v1")

    test_embedding = embeddings.embed_query("Hello world!")
    print(f"Embedding dimension: {len(test_embedding)}")
    print(f"Values: {test_embedding}")
