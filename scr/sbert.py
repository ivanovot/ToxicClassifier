from transformers import AutoTokenizer, AutoModel
import torch

# Определяем доступное устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mean Pooling - Учитывает attention mask для корректного усреднения
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # Эмбеддинги токенов
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

# Загрузка модели и токенизатора с HuggingFace
tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_mt_nlu_ru")
sbert = AutoModel.from_pretrained("ai-forever/sbert_large_mt_nlu_ru").to(device)  # Перенос модели на устройство

def vectorize(texts, batch_size=32):
    if isinstance(texts, str):
        texts = [texts]  # Если передана строка, оборачиваем её в список
    
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded_input = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            model_output = sbert(**encoded_input)
        batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).cpu()
        embeddings.append(batch_embeddings)

    # Конкатенируем батчи и убираем лишнее измерение
    return torch.cat(embeddings, dim=0)

