import requests

API_KEY = "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"  
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def generate_answer(question, context):
    prompt = (
        f"You are a helpful assistant. Use the context below to answer the question.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.7,
            "return_full_text": False
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        print(f" Error: Status code {response.status_code}")
        print(f" Response: {response.text}")
        return f"Error: {response.status_code} - {response.reason}"

    result = response.json()

    if isinstance(result, list) and "generated_text" in result[0]:
        return result[0]["generated_text"].strip()
    else:
        return " Could not parse response."
