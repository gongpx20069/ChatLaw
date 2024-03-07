from chromadb import Documents, EmbeddingFunction, Embeddings
import chromadb
from sentence_transformers import SentenceTransformer
from pprint import pprint
import json

class M3EEmbeddingFunction(EmbeddingFunction):
    embedding_model = SentenceTransformer('moka-ai/m3e-base')
    def __call__(self, input: Documents) -> Embeddings:
        batch_embeddings = self.embedding_model.encode(input)
        return batch_embeddings.tolist()

class ChineseLawDBTool:
    embed_fn = M3EEmbeddingFunction()

    def __init__(self, path="./chromadb") -> None:
        self.db_client = chromadb.PersistentClient(path=path)
        self.collection = self.db_client.get_collection(
            name=f"chinese-law",
            embedding_function=self.embed_fn,
        )
        self.description = {
            "type": "function",
            "function": {
                "name": "chinese_law_search",
                "description": "一个中国法律智能搜索工具，可以快速搜索和你输入信息最相关的详细法律条文。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "你想要查询的相关法律的信息，总结后的用户求助内容。",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def __call__(self, query:str, n:int=3):
        results = self.collection.query(
            query_texts=[query],
            n_results=n
        )
        results = [{"document":results["documents"][0][i], "cosine_similarity":results["distances"][0][i], "document_source":results["metadatas"][0][i]} for i in range(n)]
        results.sort(key=lambda x:x["cosine_similarity"])
        return json.dumps(results, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    chinese_law_db_tool = ChineseLawDBTool()
    pprint(chinese_law_db_tool("抢劫"))