# 引入 OpenAI 支持库
from langchain_openai import ChatOpenAI
# 连接信息
# 自己的vllm部署的 服务地址
base_url = "http://0.0.0.0:7860/v1"
# 自己部署的模型暂无
api_key = "xxxx"
# model 名称
model_name = "Qwen-VL-Chat"

# 连接大模型
llm = ChatOpenAI(base_url=base_url,
                api_key=api_key,
                model="qwen-vl-chat",
                temperature=0.01,
                max_tokens=512)


# 获取自定以模型连接对象
def get_self_model_connect(base_url=None,api_key=None,model_name=None,stream_option=None,
                           max_tokens=512, frequency_penalty=1.2, top_p=0.9, temperature=0.01):
    """
    
    """
    # 连接大模型
    llm = ChatOpenAI(base_url=base_url,
                    api_key=api_key,
                    model=model_name,
                    stream_options={"include_usage": stream_option}, 
                    max_tokens=512,
                    frequency_penalty=frequency_penalty, 
                    top_p=top_p,                                
                    temperature=0.01
                    )
    

    return llm


# 抽取配置文件并读取

