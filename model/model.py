import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn

class PowerfulBinaryTextClassifier(nn.Module):
    def __init__(self, model_name, lstm_hidden_size=256, num_layers=3, dropout_rate=0.2):
        super(PowerfulBinaryTextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        
        # Добавляем несколько LSTM слоев с большим размером скрытого состояния
        self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size, 
                            hidden_size=lstm_hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            bidirectional=True, 
                            dropout=dropout_rate if num_layers > 1 else 0)
        
        # Полносвязный блок с увеличенным количеством нейронов и слоев Dropout
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 2),  # полносвязный слой
            nn.Sigmoid()
        )
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)  # Инициализация токенизатора
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_outputs = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size)
        
        # Применяем LSTM
        lstm_out, _ = self.lstm(bert_outputs)  # (batch_size, sequence_length, lstm_hidden_size * 2)
        
        # Берем выход последнего временного шага для классификации
        last_time_step = lstm_out[:, -1, :]  # (batch_size, lstm_hidden_size * 2)
        
        logits = self.fc(last_time_step)  # Применяем полносвязный блок
        
        logits[:, 1] -= 0.995  # Умножаем логит для выбранного класса
        
        return logits  # Возвращаем логиты для двух классов

    def predict(self, text):
        self.to(self.device)  # Переносим модель на выбранное устройство
        
        # Токенизация текста
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        input_ids = inputs['input_ids'].to(self.device)  # Переносим на устройство
        attention_mask = inputs['attention_mask'].to(self.device)  # Переносим на устройство
        
        # Получение предсказания
        self.eval()  # Переключаем модель в режим оценки
        with torch.no_grad():
            preds = self(input_ids, attention_mask)  # Получаем логиты
        
        # Возвращаем индекс класса с наибольшей вероятностью
        return torch.argmax(preds, dim=1).item()  # Возвращаем индекс класса

    def load_weights(self, filepath):
        # Загрузка весов модели
        self.load_state_dict(torch.load(filepath, map_location=self.device, weights_only=True))

# Пример инициализации модели
model_name = "DeepPavlov/rubert-base-cased"
model = PowerfulBinaryTextClassifier(model_name)

model.load_weights('model.pth')
