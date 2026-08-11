"""
Day 1–3 exploration scratch (tokenization / embeddings / attention).
Moved out of src/main.py so FastAPI can live there again.
"""

from transformers import AutoTokenizer
from transformers import AutoModel
import torch

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

print(tokenizer.vocab_size)
print(tokenizer.model_max_length)

text = "This movie was absolutely amazing!"

tokens = tokenizer(text)
print(tokens)

input_ids = tokens["input_ids"]
print(f"Количество токенов: {len(input_ids)}")

decoded = tokenizer.decode(input_ids)
print(f"Декодировано: {decoded}")

def tokenize_texts(texts, max_length=128):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

texts = [
    "This movie was great!",
    "Terrible movie, waste of time."
]

tokens = tokenize_texts(texts)
print(f"Shape: {tokens['input_ids'].shape}")

print(f"Attention mask:\n{tokens['attention_mask']}")

print(f"CLS token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
print(f"SEP token: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

single = tokenizer(text, return_tensors="pt")
print(f"Input IDs: {single['input_ids']}")
print(f"Decoded: {tokenizer.decode(single['input_ids'][0])}")

def explain_tokenization(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)

    print(f"Исходный текст: {text}")
    print(f"Токены: {tokens}")
    print(f"IDs: {ids}")
    print(f"Количество: {len(tokens)}")

explain_tokenization("Transformers are amazing!", tokenizer)


print("day 2")
model = AutoModel.from_pretrained(model_name)

model.eval()
print(model)

text = "This movie was absolutely amazing!"
tokens = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**tokens)

print(type(outputs))
print(outputs.last_hidden_state.shape)

cls_embedding = outputs.last_hidden_state[:, 0, :]
print(f"CLS embedding shape: {cls_embedding.shape}")
print(f"CLS embedding: {cls_embedding[0][:5]}...")

def get_embeddings(texts, tokenizer, model, batch_size=32):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**tokens)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu().numpy())

    import numpy as np

    return np.vstack(all_embeddings)


texts = [
    "This movie was absolutely amazing!",
    "Terrible movie, waste of time.",
    "Pretty good, I liked it.",
    "Boring and too long.",
]

embeddings = get_embeddings(texts, tokenizer, model)
print(f"Embeddings shape: {embeddings.shape}")
print("Ожидается: (4, 768) для DistilBERT")

from sklearn.metrics.pairwise import cosine_similarity


def similarity(text1, text2, tokenizer, model):
    emb = get_embeddings([text1, text2], tokenizer, model)
    sim = cosine_similarity(emb[0:1], emb[1:2])[0][0]
    return sim


sim1 = similarity("Great movie!", "Amazing film!", tokenizer, model)
sim2 = similarity("Great movie!", "Terrible film!", tokenizer, model)

print(f"Сходство похожих: {sim1:.3f}")
print(f"Сходство разных: {sim2:.3f}")

print("day 3")
model = AutoModel.from_pretrained(model_name, output_attentions=True)
model.eval()

text = "The amazing movie won many awards"
tokens = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**tokens)

print(type(outputs.attentions))
print(f"Количество слоёв: {len(outputs.attentions)}")
print(f"Форма attention для слоя 0: {outputs.attentions[0].shape}")

import matplotlib.pyplot as plt
import seaborn as sns


def visualize_attention(tokens, attention, layer=0, head=0):
    attn = attention[layer][0, head]
    token_list = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn.cpu().numpy(),
        xticklabels=token_list,
        yticklabels=token_list,
        cmap="viridis",
        cbar=True,
    )
    plt.title(f"Attention - Layer {layer}, Head {head}")
    plt.xlabel("Keys")
    plt.ylabel("Queries")
    plt.tight_layout()
    plt.savefig(f"attention_layer{layer}_head{head}.png")
    plt.close()


visualize_attention(tokens, outputs.attentions, layer=0, head=0)
visualize_attention(tokens, outputs.attentions, layer=3, head=0)
visualize_attention(tokens, outputs.attentions, layer=5, head=0)
