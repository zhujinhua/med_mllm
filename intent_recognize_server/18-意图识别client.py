import requests
import json

# response = requests.post(
#     "http://localhost:8001/intention_rec/invoke",
#     json={"input": {"role": "意图识别专家", "content": "抗癌药物有哪些？"}}
# )
# print(json.loads(response.json()['output']['content']))

response = requests.post(
    "http://direct.virtaicloud.com:42383/intention_rec/invoke",
    json={"input": {"role": "意图识别专家", "content": "抗癌药物有哪些？"}}
)
print(json.loads(response.json()['output']['content']))

