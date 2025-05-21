import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import json


class PWLayer(nn.Module):
    """Single Parametric Whitening Layer
    """
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)
    

class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced Adaptor
    """
    def __init__(self, n_exps=8, layers=[64, 4096], dropout=0.2, noise=True):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training) # (B, n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)


class Explainer(torch.nn.Module):
    def __init__(self, token_size=4096, user_embed_size=64, item_embed_size=64):
        super(Explainer, self).__init__()
        
        # Initialize Ollama client
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model_name = "llama2"

        # add special tokens for user and item embeddings
        self.special_tokens = ["<USER_EMBED>", "<ITEM_EMBED>", "<EXPLAIN_POS>", "<pad>"]
        
        self.user_embedding_converter = MoEAdaptorLayer(n_exps=8, layers=[user_embed_size, token_size], dropout=0.2, noise=True)
        self.item_embedding_converter = MoEAdaptorLayer(n_exps=8, layers=[item_embed_size, token_size], dropout=0.2, noise=True)

    def get_ollama_response(self, prompt):
        response = requests.post(
            self.ollama_url,
            json={"model": self.model_name, "prompt": prompt},
            stream=True
        )
        generated_text = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "response" in data:
                    generated_text += data["response"]
        return generated_text

    def forward(self, user_embedding, item_embedding, input_text):
        # Convert embeddings
        converted_user_embedding = self.user_embedding_converter(user_embedding).half()
        converted_item_embedding = self.item_embedding_converter(item_embedding).half()

        # Construct prompt with embeddings
        prompt = f"User embedding: {converted_user_embedding}\nItem embedding: {converted_item_embedding}\nInput text: {input_text}"
        
        # Get response from Ollama
        response = self.get_ollama_response(prompt)
        
        return response

    def generate(self, user_embedding, item_embedding, input_text, max_new_tokens=128):
        # Convert embeddings
        converted_user_embedding = self.user_embedding_converter(user_embedding).half()
        converted_item_embedding = self.item_embedding_converter(item_embedding).half()

        # Construct prompt with embeddings
        prompt = f"User embedding: {converted_user_embedding}\nItem embedding: {converted_item_embedding}\nInput text: {input_text}"
        
        # Get response from Ollama with max tokens
        response = self.get_ollama_response(prompt)
        
        return response
        