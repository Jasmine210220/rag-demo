# RAG Demo Tutorial

一个面向初学者的 RAG 入门项目，使用 `LangChain + Chroma + Qwen` 完成两件事：

1. 把本地 CSV 知识库写入向量数据库
2. 在检索到相关内容后，把检索结果和用户问题一起交给大模型生成最终答案

这个仓库适合用来理解最基础的 RAG 在线流程，也适合用来学习 `LangChain` 中常见的链式写法，例如：

- `ChatPromptTemplate`
- `StrOutputParser`
- `RunnableLambda`
- `RunnablePassthrough`

## 1. 项目目标

这个项目想解决的问题很简单：

- 普通大模型只根据参数记忆回答问题，容易胡编
- RAG 会先从你自己的资料里检索相关内容
- 再把这些内容作为上下文交给模型回答
- 这样回答会更贴近你的知识库内容

你可以把它理解成：

`用户问题 -> 向量检索 -> 找到相关资料 -> 把资料交给模型 -> 输出答案`

## 2. 项目结构

```text
rag_demo/
├─ data/
│  └─ test_large.csv
├─ 向量存储.py
├─ rag在线流程.py
├─ README.md
└─ .gitignore
```

各文件作用如下：

- `data/test_large.csv`
  用来测试的知识库数据，包含较多条记录，便于观察向量检索效果。

- `向量存储.py`
  演示如何把 CSV 数据加载为 `Document`，再写入 Chroma 向量库，并执行一次相似度检索。

- `rag在线流程.py`
  演示完整的 RAG 在线问答流程，并且使用 `LangChain` 的链式写法，把“问题传递”和“向量检索”一起写入链中。

## 3. 运行前准备

在运行项目之前，你需要准备以下环境：

### 3.1 Python 环境

建议使用 Python 3.10 及以上版本。

### 3.2 安装依赖

至少需要这些包：

```powershell
pip install langchain-chroma langchain-community chromadb
```

如果你的环境里还没有 Tongyi / DashScope 相关依赖，也需要一并安装。

### 3.3 配置模型 API Key

本项目默认使用：

- 向量模型：`DashScopeEmbeddings`
- 对话模型：`qwen-max`

因此你需要提前在系统环境变量中配置好对应的 API Key。  
本仓库代码默认直接读取系统环境变量，不需要在脚本中手写密钥。

## 4. 测试数据说明

本项目统一使用：

```text
data/test_large.csv
```

这份 CSV 是一个较大的测试知识库，包含：

- `source`
- `title`
- `category`
- `content`

其中：

- `source` 用来标识每条数据的来源
- `content` 是最主要的文本内容
- 其余字段也会一起进入 `Document` 文本中，参与向量化和检索

## 5. 第一步：理解向量存储脚本

文件：

`向量存储.py`

这个脚本负责做以下几件事：

1. 读取 `data/test_large.csv`
2. 把 CSV 内容加载成 `Document`
3. 使用 `DashScopeEmbeddings` 进行向量化
4. 把向量数据写入 Chroma
5. 执行一次相似度检索并打印结果

运行命令：

```powershell
python .\向量存储.py
```

如果运行成功，你会看到检索结果被打印出来。



## 6. 第二步：理解在线 RAG 流程

文件：

`rag在线流程.py`

这个脚本是在向量检索的基础上再往前走一步，完成完整问答流程。

它的执行逻辑是：

1. 连接向量库
2. 检查向量库中是否已有数据
3. 如果没有数据，就自动导入 `data/test_large.csv`
4. 接收用户问题
5. 用链式方式把“原始问题”和“检索结果”一起传入提示词模板
6. 调用 `qwen-max` 生成最终回答

运行命令：

```powershell
python .\rag在线流程.py
```

## 7. RAG 在线流程图

```text
用户提出问题
    ↓
问题进入 RAG 链
    ↓
一路原样保留为 question
    ↓
一路进入向量检索，生成 context
    ↓
question 和 context 一起进入 prompt
    ↓
交给 qwen-max 生成答案
    ↓
输出最终结果
```

这就是最基础的链式 RAG。

## 8. 现在这版 RAG 为什么更像 LangChain

如果只是手动写：

1. 先 `similarity_search`
2. 再手动拼 `context`
3. 最后 `model.invoke(...)`

那虽然也能跑通，但更像“普通 Python 逻辑 + 调模型”。

现在这版代码把这些步骤进一步组织成了链，核心写法是：

```python
rag_chain = (
    {
        "question": RunnablePassthrough(),
        "context": retriever,
    }
    | prompt
    | model
    | output_parser
)
```

这样做的好处是：

- 链的结构更清晰
- 每一步职责更明确
- 更贴近 LangChain 推荐的 RAG 组合思路
- 后面继续扩展 memory、history、retriever 都更自然

## 9. RunnablePassthrough 的作用和用法

这是这次改造里最关键的一个点。

### 9.1 它是干什么的

`RunnablePassthrough()` 的作用可以简单理解为：

**把输入原样传下去，不做修改。**

例如：

```python
RunnablePassthrough()
```

如果输入是：

```python
"什么是 embedding？"
```

那么它输出的还是：

```python
"什么是 embedding？"
```

### 9.2 在这个项目里它解决了什么问题

在 RAG 中，一个用户问题通常要同时做两件事：

1. 作为检索条件去查向量库
2. 作为最终提示词里的“用户问题”交给模型

也就是说，**同一个输入要拆成两路**。

在本项目里，这两路分别是：

- `question`
  原样保留用户问题

- `context`
  用这个问题去做向量检索，再把结果整理成上下文

对应的链式写法是：

```python
{
    "question": RunnablePassthrough(),
    "context": retriever,
}
```

这里的意思是：

- `question` 这一路直接保留原始输入
- `context` 这一路把原始输入交给 `retriever`

### 9.3 retriever 在这里做了什么

本项目里写的是：

```python
retriever = RunnableLambda(
    lambda question: format_docs(
        vector_store.similarity_search(question, k=TOP_K)
    )
)
```

这说明：

- 输入一个问题
- 去向量库做相似度检索
- 取回最相关的几条文档
- 再调用 `format_docs()` 拼成字符串上下文

最后链会自动把两路结果合并成：

```python
{
    "question": "用户原问题",
    "context": "检索出来的参考资料"
}
```

再一起交给 `prompt`。

### 9.4 为什么不用普通函数就好

当然也可以只用普通函数先查，再把结果传给模型。  
但是使用 `RunnablePassthrough + RunnableLambda` 的优势是：

- 代码更贴近 LangChain 的链式思想
- 数据流更直观
- 更方便后续继续拼接别的步骤




## 10. 这两个脚本的区别

### `向量存储.py`

重点是：

- 学习如何加载数据
- 学习如何建立向量库
- 学习如何做相似度检索

它更偏“向量数据库入门”。

### `rag在线流程.py`

重点是：

- 检索只是中间步骤
- 最终目标是把检索结果交给大模型回答
- 并且把这套过程写成 LangChain 链

## 11. 运行时可能遇到的问题

### 11.1 API Key 未配置

如果环境变量没有配好，模型调用和向量化都会失败。

### 11.2 网络无法访问模型服务

如果当前网络无法访问 DashScope / Tongyi 接口，脚本会在向量化或模型调用时报错。

### 11.3 重复导入数据

`rag在线流程.py` 中做了一个“如果向量库已有数据就跳过导入”的判断，这是为了避免每次运行都重复写入同一份 CSV。


## 12. 总结

这个仓库做的事情可以概括为两步：

1. 先把知识库内容向量化并存入 Chroma
2. 再把检索结果交给 `qwen-max` 生成最终答案

而在最新版本里，这个过程已经进一步写成了 LangChain 风格的链式 RAG：

- `RunnablePassthrough` 保留原始问题
- `RunnableLambda` 完成检索
- `ChatPromptTemplate` 组织提示词
- `ChatTongyi` 调用模型
- `StrOutputParser` 输出最终字符串结果

这个项目已经覆盖了RAG最核心的一条最小闭环：

`本地数据 -> 向量存储 -> 相似度检索 -> 检索增强生成 -> 链式调用`
