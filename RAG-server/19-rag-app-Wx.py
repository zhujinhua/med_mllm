from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.embeddings import XinferenceEmbeddings
import chromadb
# from langchain.chat_models import ChatOpenAI
from model_utils.model_connect_util import model_connect_util
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List, Dict
import os

from langserve import add_routes

app = FastAPI()

# 1. 连接数据库
# 获取embeddings模型 - bge
embed = XinferenceEmbeddings(
    server_url="http://direct.virtaicloud.com:28511", model_uid="custom-bge-m3"
)

chroma_client = chromadb.HttpClient(host='direct.virtaicloud.com', port=20994)

# 自定义embedding函数
class MyEmbeddingFunction:
    def __call__(self, input: str) -> list[float]:
        # embed the documents somehow
        print("------执行了--------")
        embeddings = embed.embed_documents([input])
        return embeddings[0]

# 获取集合
collection = chroma_client.get_or_create_collection(name="testDB4", embedding_function=MyEmbeddingFunction())

# 2. 引入大模型
def get_self_model_connect(base_url=None, api_key=None, model_name=None, stream_option=None):
    """
    获取自定义模型连接对象
    """
    # 连接大模型
    # llm = ChatOpenAI(
    #     base_url=base_url,
    #     openai_api_key=api_key,
    #     model_name=model_name,
    #     temperature=0.01,
    #     max_tokens=512,
    #     streaming=stream_option
    # )
    return llm
base_url_llm = "http://direct.virtaicloud.com:25933/v1"
base_url_llm = "http://direct.virtaicloud.com:25933/v1"

llm = model_connect_util.get_self_model_connect(
    base_url=base_url_llm,
    api_key="XX",
    model_name="qwen-vl-chat",
    stream_option=True
)

# 3. 生成多种提问方式
def generate_variations(question: str) -> List[str]:
    role_description = "你是一个专业的信息重组专家，你的任务是帮助用户提供多种不同的提问方式，以便从不同的角度获取信息。"
    task_description = "请根据提供的问题，给出至少3种不同的提问方式。每种提问方式都应该尽量覆盖问题的不同方面，同时保持问题的主旨不变。"
    template = f"""{role_description}
    {task_description}
    问题：{question}
    不同的提问方式：
    1. 
    2. 
    3. 
    """
    prompt = PromptTemplate(template=template, input_variables=["question"])
    chain = LLMChain(llm=llm, prompt=prompt)
    variations = chain.run(question=question)
    variations_list = [v.strip() for v in variations.split('\n') if v.strip()]
    return variations_list

def ask_rag_system(question: str) -> List[dict]:
    query_embed = embed.embed_query(question)
    results = collection.query(query_embeddings=query_embed, n_results=1)
    parsed_results = []

    if 'documents' in results and 'metadatas' in results and 'distances' in results:
        for doc, meta, distance in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
            parsed_result = {
                'distance': distance,
                'metadata': meta,
                'document': doc
            }
            parsed_results.append(parsed_result)

    return parsed_results

def deduplicate_and_sort_results(results: List[List[Dict[str, any]]]) -> List[Dict[str, any]]:
    flattened_results = [item for sublist in results for item in sublist]
    unique_results_dict = {}
    for result in flattened_results:
        if result['distance'] > 0.85:
            continue
        metadata_key = (result['metadata']['source'], result['metadata'].get('title', ''))
        if metadata_key not in unique_results_dict:
            unique_results_dict[metadata_key] = result
        else:
            existing_result = unique_results_dict[metadata_key]
            existing_result['document'] += '\n' + result['document']
    unique_results = list(unique_results_dict.values())
    unique_results.sort(key=lambda x: x['distance'], reverse=True)
    return unique_results

# def generate_answer(question: str, answers: str) -> str:
#     template = """问题：{question}
#     答案：{answers}
#     """
#     prompt = PromptTemplate(template=template, input_variables=["question", "answers"])
#     print(prompt)
#     chain = LLMChain(llm=llm, prompt=prompt)
#     answer = chain.run(question=question, answers=answers)
#     return answer


from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
class MyVectorStoreRetriever(BaseRetriever):
    """
    基于向量数据库的 Retriever 实现
    """

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """Retriever 的同步实现"""

        variations = generate_variations(query)
        answers_lists = [ask_rag_system(variation) for variation in variations]
        aggregated_answers = deduplicate_and_sort_results(answers_lists)
        print(aggregated_answers)
        return aggregated_answers


from langchain_core.prompts import ChatPromptTemplate
# 可执行的占位符
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
  ("human", """
            你是一个医疗领域的助手。
            请使用下面检索到的药品信息来回答用户的问题。
            请辨别上下文中内容和问题是否是否相关。不相关的内容请不要回答。
            如果在提供的上下文中找不到相关信息，请直接回答“暂没有对应的药品”。
            Question: {question} 
            Context: {context} 
            Answer:

""")
])


myRetriever = MyVectorStoreRetriever()
rag_chain = (
    {"context": myRetriever,
     "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

add_routes(
    app=app,
    runnable=rag_chain,
    path="/rag_asg",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)