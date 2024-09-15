import json
import re

def decode_unicode(s):
    """解码字符串中的 Unicode 转义序列"""
    return s.encode().decode('unicode_escape')

def transform_annotation_mllm(annotation):
    # 转换为 mllm 格式
    image_id = annotation.get("image_id")
    caption = annotation.get("caption")

    return {
        "messages": [
            {
                "content": "根据X射线图像，分析心脏和肺部的情况",
                "role": "user"
            },
            {
                "content": caption,
                "role": "assistant"
            }
        ],
        "images": [
            f"images2/{image_id}.png"
        ]
    }

def transform_annotation_swift(annotation):
    # 转换为 swift 格式
    image_id = annotation.get("image_id")
    caption = annotation.get("caption")

    return {
        "imageId": image_id,
        "caption": caption,
        "filePath": f"images2/{image_id}.png"
    }

def process_and_save(input_file, mllm_file, swift_file):
    # 读取源 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # 解码 Unicode 转义序列
    if isinstance(data, str):
        data = decode_unicode(data)
        data = json.loads(data)

    # 确保 annotations 是一个数组（列表）
    annotations = data.get("annotations", [])
    if not isinstance(annotations, list):
        annotations = [annotations]  # 将单个对象转换为列表

    # 使用列表推导式处理每个 annotation
    transformed_mllm_annotations = [transform_annotation_mllm(annotation) for annotation in annotations]
    transformed_swift_annotations = [transform_annotation_swift(annotation) for annotation in annotations]

    # 保存 mllm 格式的数据到文件
    with open(mllm_file, 'w', encoding='utf-8') as file:
        json.dump(transformed_mllm_annotations, file, ensure_ascii=False, indent=2)

    # 保存 swift 格式的数据到文件
    with open(swift_file, 'w', encoding='utf-8') as file:
        json.dump(transformed_swift_annotations, file, ensure_ascii=False, indent=2)

    print(f"mllm 格式数据已保存到 {mllm_file}")
    print(f"swift 格式数据已保存到 {swift_file}")

# 处理数据并保存到文件
process_and_save('openi-zh.json', 'mllm_data.json', 'swift_data.json')
