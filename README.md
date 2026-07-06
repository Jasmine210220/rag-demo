# RAG Demo

这是一个简单的 RAG 学习示例，包含两个脚本：

- `向量存储.py`：把 CSV 数据写入 Chroma，并做一次相似度检索
- `rag在线流程.py`：先做向量检索，再把检索结果和用户问题一起交给 `qwen-max` 生成最终答案

## 目录结构

```text
rag_demo/
├─ data/
│  └─ test_large.csv
├─ 向量存储.py
├─ rag在线流程.py
└─ .gitignore
```

## 运行前准备

1. 安装依赖：`langchain-chroma`、`langchain-community`、`chromadb`
2. 系统环境变量中已经配置好 DashScope / Tongyi 对应的 API Key

## 运行方式

先执行向量存储脚本：

```powershell
python .\向量存储.py
```

再执行在线 RAG 流程：

```powershell
python .\rag在线流程.py
```

## 说明

- 两个脚本都使用脚本所在目录下的相对路径
- 向量库存储目录已经独立，便于单独上传到 GitHub
- `.gitignore` 默认忽略本地生成的向量库文件和缓存文件

