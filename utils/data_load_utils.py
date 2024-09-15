import pandas as pd
import json
import uuid

from langchain_core.documents import Document


def json_to_str(data=None):
    """
        json data to str
        :param data: str data
    """
    docs = []
    for doc in data:
        temp = doc
        doc = str(doc)
        docs.append(doc)
    return docs

def str_to_document(data=None):
    """
        str data to Document object
        :param data: str data
    """
    docs = []
    for doc in data:
        temp = doc
        doc = str(doc)
        doc = Document(page_content=doc,metadata=dict(source=temp['标题链接'],title=temp['标题']))
        docs.append(doc)
    return docs

def excel_to_json(excel_file=None,sheet_name='Sheet1'):
    """
    excel data to json
    :param excel_file: Excel文件路径
    :param sheet_name: Sheet名称 -预留
    """
    # 读取Excel文件
    xls = pd.ExcelFile(excel_file)
    
    # 创建一个空字典存储每个Sheet的数据
    excel_data = {}
    
    # 遍历每个Sheet
    for sheet_name in xls.sheet_names:
        # 读取Sheet内容为DataFrame
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        # 将DataFrame转换为字典格式
        data_dict = df.to_dict(orient='records')
        
        # 将数据存入字典，以Sheet名为键
        excel_data[sheet_name] = data_dict
    
    # 将字典转换为JSON字符串
    # json_data = json.dumps(excel_data, indent=4, ensure_ascii=False)
    
    # return json_data
    return excel_data

def get_ids(type="uuid"):
    """
        获取uuid
        :param type: 获取id类型
        :return: id
        :description: https://www.jb51.net/python/325331nfi.htm
    """
    return str(uuid.uuid4()).replace("-","")
