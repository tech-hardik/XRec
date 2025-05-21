import argparse
import torch
from models.explainer import Explainer
import json
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='generate', choices=['finetune', 'generate'])
    parser.add_argument('--dataset', type=str, default='amazon', choices=['amazon', 'google', 'yelp'])
    args = parser.parse_args()

    # Initialize device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load data
    data_path = f"data/{args.dataset}"
    with open(os.path.join(data_path, "data.json"), "r") as f:
        data = json.load(f)
    
    # Initialize model
    model = Explainer().to(device)
    
    if args.mode == 'finetune':
        print("Finetuning mode is not needed with Ollama as we're using the model directly")
        return
    
    elif args.mode == 'generate':
        print("Generating explanations...")
        # Load user and item embeddings
        user_emb = torch.load(os.path.join(data_path, "user_emb.pkl"))
        item_emb = torch.load(os.path.join(data_path, "item_emb.pkl"))
        
        # Generate explanations
        explanations = []
        for item in data:
            user_embedding = user_emb[item['uid']]
            item_embedding = item_emb[item['iid']]
            input_text = f"User profile: {item['user_profile']}\nItem profile: {item['item_profile']}"
            
            explanation = model.generate(user_embedding, item_embedding, input_text)
            explanations.append({
                'uid': item['uid'],
                'iid': item['iid'],
                'explanation': explanation
            })
        
        # Save explanations
        with open(os.path.join(data_path, "tst_pred.pkl"), "wb") as f:
            torch.save(explanations, f)
        print("Explanations generated and saved!")

if __name__ == "__main__":
    main()
