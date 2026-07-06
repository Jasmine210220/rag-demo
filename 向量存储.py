from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import DashScopeEmbeddings


BASE_DIR = Path(__file__).resolve().parent

# 统一使用较大的测试 CSV 文件
CSV_PATH = BASE_DIR / "data" / "test_large.csv"

# 当前脚本使用的向量库存储路径
PERSIST_DIRECTORY = BASE_DIR / "chroma_db"


# 创建向量存储对象
vector_store = Chroma(
    collection_name="test",                    # 当前向量集合名称
    embedding_function=DashScopeEmbeddings(),  # 使用的向量模型
    persist_directory=str(PERSIST_DIRECTORY),  # 向量数据库保存位置
)

# 把 CSV 文件加载成 Document 对象列表
loader = CSVLoader(
    file_path=str(CSV_PATH),
    encoding="utf-8",
    source_column="source",                    # 指定 source 这一列作为文档来源
)

documents = loader.load()

# 把 Document 列表转成向量并写入 Chroma
vector_store.add_documents(
    documents=documents,
    ids=[f"id{i}" for i in range(len(documents))],  # 给每条数据分配唯一 id
)

# 用一个测试问题做相似度检索
result = vector_store.similarity_search(
    "embedding是什么意思？",
    3,
)

print(result)
