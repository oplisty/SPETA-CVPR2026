# 首先，确保安装了必要的库：
# pip install langchain langchain-community langchain-openai beautifulsoup4 chromadb

import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 设置您的API密钥，例如OpenAI
os.environ["OPENAI_API_KEY"] = "您的-OpenAI-API-KEY"

# 2. 指定要抓取的网页URL
url = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
loader = WebBaseLoader(url)
docs = loader.load()

# 3. 将长文档分割成小块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 4. 将文本块向量化并存入向量数据库
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# 5. 创建检索器
retriever = vectorstore.as_retriever()

# 6. 定义提示模板（使用LangChain Hub上的一个流行提示）
prompt = hub.pull("rlm/rag-prompt")

# 7. 定义LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# 8. 组装完整的RAG流水线
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 9. 提问！
question = "What is RAG?"
answer = rag_chain.invoke(question)
print(f"问题: {question}")
print(f"答案: {answer}")