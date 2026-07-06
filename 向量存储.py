from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import DashScopeEmbeddings


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "test_large.csv"
PERSIST_DIRECTORY = BASE_DIR / "chroma_db"


# 创建文本转向量对象
vector_store = Chroma(
    collection_name="test",                     # 当前向量存储名称
    embedding_function=DashScopeEmbeddings(),   # 使用的文本转向量模型
    persist_directory=str(PERSIST_DIRECTORY),   # 指定向量库存放路径
)

# 先把CSV文档转成Document对象
loader = CSVLoader(
    file_path=str(CSV_PATH),
    encoding="utf-8",
    source_column="source",                     # 指定每条数据来源列
)

documents = loader.load()

# 把Document列表对象转成向量数据并存入Chroma
vector_store.add_documents(
    documents=documents,
    ids=[f"id{i}" for i in range(len(documents))],   # 给每条数据一个唯一id
)

# 查询数据
result = vector_store.similarity_search(
    "embedding是什么意思？",
    3,
)

print(result)

