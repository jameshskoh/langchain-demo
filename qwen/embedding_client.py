import requests
from langchain_core.embeddings import Embeddings


class QwenEmbeddings(Embeddings):

    def __init__(self, base_url: str):
        self.base_url = base_url

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embedding_endpoint = f"{self.base_url}/embeddings"
        response = requests.post(
            embedding_endpoint,
            json={"input": texts, "model": "text-embedding-qwen3-embedding-4b"},
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer dummy",
            },
        )

        if response.status_code == 200:
            data = response.json()
            return [item["embedding"] for item in data["data"]]
        else:
            raise Exception(f"Failed to get embeddings: {response.text}")

    def embed_query(self, text: str) -> list[float]:
        response = requests.post(
            f"{self.base_url}/embeddings",
            json={
                "input": [text],  # Send as array with single string
                "model": "text-embedding-qwen3-embedding-4b",
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer dummy",
            },
        )
        if response.status_code == 200:
            data = response.json()
            return data["data"][0]["embedding"]
        else:
            raise Exception(f"Failed to get embedding: {response.text}")
