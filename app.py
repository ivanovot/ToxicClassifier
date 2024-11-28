import gradio as gr
from scr.model import Model
import torch

# Настройка устройства и загрузка модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Model()
model.load_state_dict(torch.load('models/model_epoch_50.pt', map_location=device))
model.eval()

# Функция для предсказания оценки токсичности текста
def predict_text(text):
    # Проверяем длину текста
    word_count = len(text.split())
    if word_count < 3:
        return "Слишком короткий текст", None
    
    # Предсказываем результат
    score = round(float(model.predict(text).item()), 5)  # Приводим результат к числу с 5 знаками после запятой
    return f"Оценка токсичности: {score}", score

# Примеры для демонстрации
examples = [
    "Этот продукт просто великолепен, спасибо!",
    "Ты ужасен, не могу терпеть твои комментарии!",
    "Сегодня был хороший день, несмотря на небольшой дождь.",
    "Твой проект провалился, и это только твоя вина.",
    "Замечательная работа, вы молодцы!"
]

# Создаем интерфейс
demo = gr.Interface(
    fn=predict_text,  # Функция для предсказания
    inputs=gr.Textbox(
        label="Введите текст для проверки на токсичность",  # Подпись для текстового поля
        placeholder="Напишите комментарий для анализа",     # Подсказка для ввода
        lines=5,                                             # Количество строк
        interactive=True                                     # Включаем интерактивность
    ),
    outputs=[
        gr.Textbox(
            label="Результат анализа",  # Подпись для вывода
            placeholder="Оценка токсичности будет показана здесь",  # Подсказка для вывода
        ),
        gr.Slider(
            label="Шкала токсичности",  # Подпись шкалы
            minimum=0.0,
            maximum=1.0,
            step=0.00001,
            interactive=False,          # Делаем слайдер только для вывода
        )
    ],
    examples=examples,  # Примеры для пользователей
    title="Toxicity Classification",  # Заголовок
    description="Введите текст, чтобы узнать его оценку токсичности (0 - не токсичный, 1 - максимально токсичный).",  # Описание
    live=True,  # Автоматический запуск модели при изменении текста
)

# Запуск приложения
demo.launch()
