import json
import requests
import numpy as np

"""This file is used to generate the explanation of user and item interactions"""

system_prompt = ""
with open("generation/explanation/exp_system_prompt.txt", "r") as f:
    system_prompt = f.read()

explanation_prompts = []
with open("generation/explanation/exp_prompts.json", "r") as f:
    for line in f.readlines():
        explanation_prompts.append(json.loads(line))

def get_ollama_response(input):
    uid = input["uid"]
    iid = input["iid"]
    prompt = json.dumps(input["feedback"], indent=4)
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
    result = {"uid": uid, "iid": iid, "explanation": generated_text}
    return result

indexs = len(explanation_prompts)
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
print(explanation_prompts[picked_id])
print("---------------------------------------------------\n")
response = get_ollama_response(explanation_prompts[picked_id])
print(Colors.GREEN + "Generated Results:\n" + Colors.END)
print(response)
