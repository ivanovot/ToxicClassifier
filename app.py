import gradio as gr
from model import model  # Импортируем вашу модель

# Токсичные и нетоксичные комментарии для тестирования
examples = [
    ["Страницу обнови, дебил. Это тоже не оскорбление, а доказанный факт - не-дебил про себя во множественном числе писать не будет. Или мы в тебя верим - это ты и твои воображаемые друзья?"],
    ["УПАД Т! ТАМ НЕЛЬЗЯ СТРОИТЬ! ТЕХНОЛОГИЙ НЕТ! РАЗВОРУЮТ КАК ВСЕГДА! УЖЕ ТРЕЩИНАМИ ПОШ Л! ТУПЫЕ КИТАЗЫ НЕ МОГУТ НИЧЕГО НОРМАЛЬНО СДЕЛАТЬ!"],
    ["хорош врать, ты террорист-торчёк-шизофреник пруф: а вот без костюма да чутка учёный, миллиардер, филантроп"],
    ["Мне Вас очень жаль, если для Вас оскорбления - норма"],
    ["Осторожней на сверхманёврах. В предыдущей методичке у вас было написано, что добрые арабы никогда ни с кем не воевали, только торговали пряностями, лел. Шапочку из фольги сними"],
    ["Так то стоит около 12,5 тысяч, но мне вышло в 6636 рублей и бесплатная доставка"],
    ["Ну хочешь я стану твоим другом? Как тебя зовут? Чем увлекаешься?"],
    ["Ну так это в плане изготовления изделий своими руками,а вот готовить вроде умею.Короче буду сам на себе испытывать божественный напиток и куплю огнетушитель (промышленный на всякий случай)."],
    ["Я согласен, что это хорошая идея! Давайте подумаем, как можно улучшить её еще больше."],
    ["Очень полезная информация, спасибо за подробное объяснение! Я многому научился."],
    ["Мне нравится, как вы объясняете! Это действительно помогает разобраться в теме."],
    ["Отлично написано, теперь я лучше понимаю, как работать с этим инструментом."],
    ["Классная идея! Надо попробовать и посмотреть, как это работает на практике."],
    ["Ваши советы очень полезны. Это точно сэкономит мне время на следующий раз."],
    ["Спасибо за ваши рекомендации! Я обязательно попробую это в будущем."],
    ["Мне нравится ваше решение этой проблемы. Очень креативно и практично."],
    ["Спасибо за помощь! Вы помогли мне разобраться в ситуации и сэкономить много времени."],
    ["Согласен с вами, это действительно важный аспект, о котором стоит задуматься."],
    ["Очень вдохновляющий пост! Я буду следовать вашим рекомендациям."],
]

# Функция для предсказания токсичности текста
def predict_text(text):
    word_count = len(text.split())
    if word_count < 7:
        return "Слишком короткий текст"
    
    output = model.predict(text)
    return "Токсичный" if output == 1 else "Не токсичный"

# Создаем интерфейс с улучшениями
demo = gr.Interface(
    fn=predict_text,  # Функция для предсказания
    inputs=gr.Textbox(
        label="Введите текст для проверки на токсичность",  # Подпись для текстового поля
        placeholder="Напишите комментарий для анализа",     # Подсказка для поля ввода
        lines=5,                                             # Количество строк для текста
        interactive=True,                                    # Интерактивность для пользователя
    ),
    outputs=gr.Textbox(
        label="Результат анализа",  # Подпись для вывода
        placeholder="Результат токсичности текста будет здесь",  # Подсказка для вывода
    ),
    live=True,  # Включаем live обновление
    examples=examples,  # Примеры для пользователей
    title="Тестирование токсичности текста",  # Заголовок интерфейса
    description="Введите любой текст, чтобы проверить его на токсичность. Модель проанализирует, является ли текст токсичным или нет.",  # Описание
)

# Запуск приложения с улучшенным интерфейсом
demo.launch(server_name="127.0.0.1", server_port=7860)
