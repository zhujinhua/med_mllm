"""
Author: jhzhu
Date: 2024/9/9
Description: 
"""
import uvicorn
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate
from langserve import add_routes

# from aigc.traning.model_connect_util import get_self_model_connect

# chat = get_self_model_connect(base_url='http://direct.virtaicloud.com:20925/v1', stream_option=False)
from model_utils import model_connect_util
# url = "http://direct.virtaicloud.com:28408/v1"
# url = "http://direct.virtaicloud.com:20326/v1"
url = "http://direct.virtaicloud.com:25933/v1"
chat = model_connect_util.get_self_model_connect(base_url=url,api_key="xx",
                                            model_name="qwen-vl-chat",
                                            stream_option=True)

app = FastAPI()


def create_intention_prompt():
    messages = [
        SystemMessagePromptTemplate.from_template(template='''
        你是一个{role}，根据用户的输入识别用户的意图。意图分类：
        1. 用户需要推荐药品（用户需要了解或查询具体的药品信息），
        2. 用户不需要查询药品。
        请参考以下Json模板输出，不要添加任何除Json格式的任何內容：
        {{
            "intention": 1,
            "reason": "用户询问了治疗感冒的常用药物，属于推荐药品范畴，应该推荐药品。"
        }}
    '''),
        HumanMessagePromptTemplate.from_template(
            template='输入: {content}'
        )
    ]
    return ChatPromptTemplate.from_messages(messages=messages)


def intention_recognition():
    prompt = create_intention_prompt()
    chain = prompt | chat
    add_routes(app, chain, path="/intention_rec")


if __name__ == "__main__":
    intention_recognition()
    uvicorn.run(app, host="0.0.0.0", port=8001)
