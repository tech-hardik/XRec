import json
import requests
import numpy as np

"""This file is used to generate the item profile, which contains the summary of the items"""

system_prompt = ""
with open("generation/item_profile/item_system_prompt.txt", "r") as f:
    system_prompt = f.read()

item_prompts = []
with open("generation/item_profile/item_prompts.json", "r") as f:
    for line in f.readlines():
        item_prompts.append(json.loads(line))

def get_ollama_response(input):
    iid = input["iid"]
    prompt = input["prompt"]
    # Construct the full prompt with system prompt
    full_prompt = f"{system_prompt}\n\n{prompt}"
    # Call Ollama API
    response = requests.post("http://localhost:11434/api/generate", json={"model": "llama2", "prompt": full_prompt}, stream=True)
    generated_text = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            if "response" in data:
                generated_text += data["response"]
    result = {"iid": iid, "business summary": generated_text}
    return result

indexs = len(item_prompts)
picked_id = np.random.choice(indexs, size=1)[0]


class Colors:
    GREEN = "\033[92m"
    END = "\033[0m"


print(Colors.GREEN + "Generating Profile for Item" + Colors.END)
print("---------------------------------------------------\n")
print(Colors.GREEN + "The System Prompt (Instruction) is:\n" + Colors.END)
print(system_prompt)
print("---------------------------------------------------\n")
print(Colors.GREEN + "The Input Prompt is:\n" + Colors.END)
print(item_prompts[picked_id])
print("---------------------------------------------------\n")
response = get_ollama_response(item_prompts[picked_id])
print(Colors.GREEN + "Generated Results:\n" + Colors.END)
print(response)
