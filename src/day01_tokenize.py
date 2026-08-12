"""Day 1 — tokenization (assignment entrypoint)."""

from src.embeddings import MODEL_NAME, tokenize_texts, tokenizer


def explain_tokenization(text, tok=None):
    tok = tok or tokenizer
    tokens = tok.tokenize(text)
    ids = tok.convert_tokens_to_ids(tokens)
    print(f"Исходный текст: {text}")
    print(f"Токены: {tokens}")
    print(f"IDs: {ids}")
    print(f"Количество: {len(tokens)}")


if __name__ == "__main__":
    print(f"model: {MODEL_NAME}")
    print(f"vocab_size: {tokenizer.vocab_size}")
    print(f"model_max_length: {tokenizer.model_max_length}")

    text = "This movie was absolutely amazing!"
    tokens = tokenizer(text)
    print(tokens)
    print(f"Количество токенов: {len(tokens['input_ids'])}")
    print(f"Декодировано: {tokenizer.decode(tokens['input_ids'])}")

    batch = tokenize_texts(
        ["This movie was great!", "Terrible movie, waste of time."]
    )
    print(f"Shape: {batch['input_ids'].shape}")
    print(f"Attention mask:\n{batch['attention_mask']}")

    print(f"CLS token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
    print(f"SEP token: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    explain_tokenization("Transformers are amazing!", tokenizer)
