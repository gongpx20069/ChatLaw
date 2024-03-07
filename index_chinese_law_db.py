# coding: utf-8
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from chromadb import Documents, EmbeddingFunction, Embeddings
import chromadb
from sentence_transformers import SentenceTransformer
import uuid
from tqdm import tqdm
from pprint import pprint
import config
from utils.tools import M3EEmbeddingFunction

print("# documents loading...")
loader = DirectoryLoader(config.LAW_DIR, glob="**/*.md", loader_cls=TextLoader, show_progress=True, loader_kwargs={"encoding":"UTF-8"})
docs = loader.load()
print("# documents loaded")

headers_to_split_on = [
    ("#", "Title"),
    ("##", "Subtitle"),
    ("###", "Article"),
    ("####", "Section"),
]

header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=500,
    chunk_overlap=20,
    length_function=len,
)

def splitMDTextToDocs(text):
    docs = header_splitter.split_text(text)
    return text_splitter.split_documents(docs)

print("# start to split law documents")

all_docs = []
for doc in docs:
    if not any(string in doc.metadata["source"] for string in config.IGNORE_MDS):
        all_docs += splitMDTextToDocs(doc.page_content)
    else:
        print(f"# ignore md: {doc.metadata['source']}")

embedding_model = SentenceTransformer('moka-ai/m3e-base')
db_client = chromadb.PersistentClient(path=config.DB_PATH)
embed_fn = M3EEmbeddingFunction()

# create collection
collection = db_client.get_or_create_collection(
    name=f"chinese-law",
    embedding_function=embed_fn,
    metadata={"hnsw:space": "cosine"}
)

print("# start to add documents into vector db, this process may need more time.")
for item in tqdm(all_docs):
    if len(item.metadata)>0:
        collection.add(
            documents = [item.page_content],
            metadatas = [item.metadata],
            ids = [str(uuid.uuid4())]
        )

print("# vector db created successfully!")
print("# test query: 抢劫判刑")
results = collection.query(
    query_texts=["抢劫判刑"],
    n_results=3
)
print("# results validated. ")