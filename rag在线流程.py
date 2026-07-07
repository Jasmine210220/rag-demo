from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


BASE_DIR = Path(__file__).resolve().parent

# 这里配置测试数据路径
CSV_PATH = BASE_DIR / "data" / "test_large.csv"

# 这里配置新的向量库存储路径，和向量存储.py 分开
PERSIST_DIRECTORY = BASE_DIR / "chroma_db_rag_demo"

# 当前向量库集合名称
COLLECTION_NAME = "rag_demo"

# 相似度检索时返回前几条最相关的数据
TOP_K = 4


def build_vector_store() -> Chroma:
    # 创建一个 Chroma 向量存储对象
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=DashScopeEmbeddings(),
        persist_directory=str(PERSIST_DIRECTORY),
    )


def load_csv_documents():
    # 先把 CSV 文件加载成 Document 对象列表
    loader = CSVLoader(
        file_path=str(CSV_PATH),
        encoding="utf-8",
        source_column="source",
    )
    return loader.load()


def ensure_documents(vector_store: Chroma) -> None:
    # 先检查当前向量库里是否已经有数据
    existing_count = vector_store._collection.count()

    # 如果已有数据，就跳过导入，避免重复写入
    if existing_count > 0:
        print(f"当前向量库已有 {existing_count} 条记录，跳过导入。")
        return

    documents = load_csv_documents()
    ids = [f"doc_{i}" for i in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=ids)
    print(f"已导入 {len(documents)} 条记录到新的向量库。")


def format_docs(documents) -> str:
    # 把检索到的多个 Document 整理成一个长字符串
    blocks = []
    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        blocks.append(f"[参考资料 {index}] source={source}\n{doc.page_content}")
    return "\n\n".join(blocks)


def build_rag_chain(vector_store: Chroma):
    # 这里定义检索函数：输入用户问题，输出格式化后的上下文
    retriever = RunnableLambda(
        lambda question: format_docs(
            vector_store.similarity_search(question, k=TOP_K)
        )
    )

    # 这里使用 ChatPromptTemplate 定义提示词模板
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个基于检索结果回答问题的助手。"
                "请优先依据提供的参考资料回答，"
                "不要编造资料中没有明确提到的事实。"
                "如果参考资料不足以回答，就直接说明资料不足。",
            ),
            (
                "human",
                "用户问题：{question}\n\n"
                "参考资料：\n{context}\n\n"
                "请用简洁、准确的中文回答，并在最后附上你主要参考的 source。",
            ),
        ]
    )

    # 创建聊天模型对象
    model = ChatTongyi(model="qwen-max")

    # 创建字符串输出解析器
    output_parser = StrOutputParser()

    # 这里把“问题原样传递”和“根据问题做检索”一起写进链里
    rag_chain = (
        {
            "question": RunnablePassthrough(),
            "context": retriever,
        }
        | prompt
        | model
        | output_parser
    )
    return rag_chain


def ask_with_rag(question: str) -> str:
    # 第一步：连接向量库
    vector_store = build_vector_store()

    # 第二步：如果向量库为空，先导入 CSV 数据
    ensure_documents(vector_store)

    # 第三步：构建完整的 RAG 链
    rag_chain = build_rag_chain(vector_store)

    # 第四步：直接把用户问题交给链执行
    response = rag_chain.invoke(question)
    return response


if __name__ == "__main__":
    user_question = "向量检索拿到结果之后，RAG 一般如何把结果交给模型生成最终答案？"
    answer = ask_with_rag(user_question)
    print("用户问题：", user_question)
    print("RAG回答：")
    print(answer)
