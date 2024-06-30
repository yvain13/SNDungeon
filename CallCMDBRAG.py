import requests
import json
print("Hello from CMDB RAG Agent")
url = "http://127.0.0.1:5000/query"
headers = {"Content-Type": "application/json"}
data = {"query": "How many different class we have in our inventory and Tell me how many items in each of them and also find the most used cost center? ?"}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response)
if response.status_code == 200:
    print("Response:", response.json()['response'])
else:
    print("Error:", response.status_code, response.text)
