import streamlit as st
from PIL import Image
import torch
from model_utils import model_connect_util_hy as model_connect_util
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate
import requests
import io
import json

from langserve import RemoteRunnable

# User and assistant names
U_NAME = "User"
A_NAME = "Assistant"


# 运行命令：
# 1. streamlit run .\13.web_demo_streamlit-3_1.py  --server.port 8001 --server.address 0.0.0.0    
# 2. streamlit run .\13.web_demo_streamlit-3_1.py

# Sidebar settings for user information (not hidden)
st.sidebar.title("大聪明药房")
st.sidebar.markdown("#### 您好，请输入您的信息。")
name   = st.sidebar.text_input('如何称呼您：')
age    = st.sidebar.text_input('您的年龄：')
gender = st.sidebar.text_input('您的性别：')
medicalhistory = st.sidebar.text_input('您是否有家族遗传病史，既往病史或过敏史：')

buttonUser = st.sidebar.button("提交", key="user")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize parameters in session state
if 'max_tokens' not in st.session_state:
    st.session_state.max_tokens = 2048

if 'frequency_penalty' not in st.session_state:
    st.session_state.frequency_penalty = 1.05

if 'top_p' not in st.session_state:
    st.session_state.top_p = 0.5

if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.7

# Sidebar settings for parameters (with expander for hiding)
with st.sidebar.expander("调参设置", expanded=False):
    max_tokens = st.slider("max_tokens", 0, 4096, st.session_state.max_tokens, step=2)
    frequency_penalty = st.slider("frequency_penalty", -2.0, 2.0, st.session_state.frequency_penalty, step=0.01)
    top_p = st.slider("top_p", 0.0, 1.0, st.session_state.top_p, step=0.01)
    temperature = st.slider("temperature", 0.0, 1.0, st.session_state.temperature, step=0.01)

    # Update session state with sidebar values
    st.session_state.max_tokens = max_tokens
    st.session_state.frequency_penalty = frequency_penalty
    st.session_state.top_p = top_p
    st.session_state.temperature = temperature

    # 获取侧边栏的参数值
    max_tokens = st.session_state.max_tokens
    frequency_penalty = st.session_state.frequency_penalty
    top_p = st.session_state.top_p
    temperature = st.session_state.temperature



# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer(max_tokens, frequency_penalty, top_p, temperature):
    print(f"load_model_and_tokenizer from {model_path}")

    url = "http://direct.virtaicloud.com:28408/v1"
    model = model_connect_util.get_self_model_connect(base_url=url, api_key="xx",
                                                      model_name="qwen-vl-chat",
                                                      stream_option=True)
   
    return model


# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = load_model_and_tokenizer(
        max_tokens = max_tokens,
        frequency_penalty = frequency_penalty,
        top_p = top_p, 
        temperature = temperature)
    print("model and tokenizer had loaded completed!")



# 解决多次加载问题
@st.cache_data
def upload_image_file_server(uploaded_image):
    url = 'http://direct.virtaicloud.com:25609/uploadfile/'

    response = requests.post(url=url, files={'file': uploaded_image})

    # Add uploaded image to chat history
    st.session_state.chat_history.append({"role": "user", "content": None, "image": response.json()["filename"]})
    print(st.session_state.chat_history)
    # display_chat_history()

# Clear chat history button
buttonClean = st.sidebar.button("清除历史消息", key="clean")
if buttonClean:
    st.session_state.chat_history = []
    st.session_state.response = ""
    # 解决cache_data方法清除历史不能再次上传远程服务器的问题
    upload_image_file_server.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    st.rerun()

# Display chat history
def display_chat_history():
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            with st.chat_message(name="user", avatar="user"):
                if message["image"] is not None:
                    response = requests.get(message["image"])
                    img_bytes = io.BytesIO(response.content)
                    st.image(image=img_bytes, caption='User uploaded image', width=448, use_column_width=False)
                    continue
                elif message["content"] is not None:
                    st.markdown(message["content"])
        else:
            with st.chat_message(name="model", avatar="assistant"):
                st.markdown(message["content"])

# Select mode
selected_mode = st.sidebar.selectbox("Select mode", ["Text", "Image"])

# 上传图片
if selected_mode == "Image":
    uploaded_image = st.sidebar.file_uploader("Upload image", key=1, type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    if uploaded_image is not None:
        upload_image_file_server(uploaded_image=uploaded_image)

# User input box
user_text = st.chat_input("Enter your question")

# 显示图片
display_chat_history()

# 构建历史消息：多轮对话
def _parse_history_msg():
    send_message = ""
    pic_count = 0
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            if message["image"] is not None:
                content = message["image"]
                pic_count += 1
                send_message += f"Picture {pic_count}:<img>{content}</img>\n "
            elif message["content"] is not None:
                content = message["content"]
                send_message += f"user: {content}\n "
        else:
            content = message["content"]
            send_message += f"assistant: {content}\n "
    send_message += f"assistant: \n"
    return send_message


def parse_event_and_data(line):
    parts = line.split(': ', 1)
    if len(parts) != 2:
        raise ValueError(f"无效的事件格式: {line}")
    event, data = parts
    return event.strip(), data.strip()

def handle_metadata(data):
    metadata = json.loads(data)
    st.markdown({metadata})

def handle_data(data):
    try:
        # 尝试解析JSON数据
        data_dict = json.loads(data)
        content = data_dict.get('content', '')
        recommendations = data_dict.get('recommendations', [])
        
        # 显示模型生成的诊断内容
        st.markdown(f"{content}", unsafe_allow_html=True)

        # 如果有药品推荐信息，生成购买按钮
        if recommendations:
            st.markdown("### 推荐药品:")
            for item in recommendations:
                st.write(f"药品名称: {item['name']}, 价格: {item['price']}, 数量: {item['quantity']}")
                if st.button(f"购买 {item['name']}", key=item['name']):
                    # 生成付款页面
                    create_payment_page(item)
    except json.JSONDecodeError:
        st.error("无法解析模型返回的数据，请检查模型的输出格式。")

def create_payment_page(item):
    st.markdown(f"### 购买药品：{item['name']}")
    st.write(f"价格: {item['price']}")
    st.write("请填写付款信息：")

    # 模拟付款表单
    with st.form(key="payment_form"):
        name = st.text_input("姓名")
        card_number = st.text_input("银行卡号")
        expiration_date = st.text_input("有效期 (MM/YY)")
        cvv = st.text_input("CVV")
        
        submit_button = st.form_submit_button(label="确认付款")

    if submit_button:
        st.success(f"付款成功！已购买 {item['name']}。")



if user_text:
    with st.chat_message(U_NAME, avatar="user"):
        st.session_state.chat_history.append({"role": "user", "content": user_text, "image": None})
        st.markdown(f"{U_NAME}: {user_text}", unsafe_allow_html=True)
    # Generate reply using the model
    model = st.session_state.model

    with st.chat_message(A_NAME, avatar="assistant"):
        if len(st.session_state.chat_history) >= 1:
            print("--------------------------------")
            for qa in st.session_state.chat_history:
                print(qa)
            print("--------------------------------")

        send_messages = _parse_history_msg()
        print(send_messages)

        # 意图识别 抗癌药物有哪些？
        print("user_text = "+user_text)
        yt_get = requests.post(
            "http://direct.virtaicloud.com:22440/intention_rec/invoke",
            json={"input": {"role": "意图识别专家", "content": user_text}}
        )
        
        #print(yt_get.text) 
        yt = 2
        try:
            yt = json.loads(yt_get.json()['output']['content'])["intention"]
        except Exception as e:
            print(e)
        
        print(yt)
        # 意图判断
        if int(yt)==1:
            url = "http://127.0.0.1:8000/rag_asg"
            # url = "http://direct.virtaicloud.com:26697/rag_asg"
            
            # 参考：https://python.langchain.com/v0.2/docs/langserve/#docs
            openai = RemoteRunnable(url)
            
            response = openai.stream(input=user_text)
        else:
            # 构建消息
            messages = []
            #医生prompt
            sys_msg = SystemMessagePromptTemplate.from_template(template="""
                        你是一个医疗专家，请根据用户的输入和历史对话，为用户进行诊疗，辅助病人进行并且判断及用药！
                        """)
            user_msg = HumanMessagePromptTemplate.from_template(template="""
                        用户的称呼:{name}
                        用户的年龄:{age}
                        用户的性别:{gender}
                        用户的家族病史:{medicalhistory}
                        用户的输入:{user_text}

                        回答用户问题时请使用用户的称呼，结合用户的年龄,用户的性别，用户的家族病史等因素综合考虑。
                        """)
            messages = [sys_msg, user_msg]
            prompt = ChatPromptTemplate.from_messages(messages=messages)

            output_parser = StrOutputParser()
            chain = prompt | model | output_parser

            # 输入信息
            input_data = {
                "name": name,
                "age": age,
                "gender": gender,
                "medicalhistory": medicalhistory,
                "user_text": send_messages
            }

            print(input_data)
            response = chain.stream(input=input_data)

            # 打印参数
            print(f"Parameters used for model request: max_tokens={st.session_state.max_tokens}, frequency_penalty={st.session_state.frequency_penalty}, top_p={st.session_state.top_p}, temperature={st.session_state.temperature}")

            print(response)
        generated_text = st.write_stream(response)

        st.session_state.chat_history.append({"role": A_NAME, "content": generated_text, "image": None})

    st.divider()


