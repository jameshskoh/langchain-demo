from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qwen.embedding_client import QwenEmbeddings


def split_text(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=80, add_start_index=True
    )
    return text_splitter.split_documents(documents)


if __name__ == "__main__":
    file_path = "../resources/nke-10k-2023.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    texts = split_text(docs)

    embeddings = QwenEmbeddings(base_url="http://localhost:1234/v1")

    # to work with qdrant instead https://python.langchain.com/docs/integrations/vectorstores/qdrant/
    vector_store = InMemoryVectorStore(embeddings)
    ids = vector_store.add_documents(documents=texts)

    results = vector_store.similarity_search("How many distribution centers does Nike have in the US?")

    print(results)

    diverse_result = vector_store.max_marginal_relevance_search("How many distribution centers does Nike have in the US?")

    print(diverse_result)
