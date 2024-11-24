from fastapi import FastAPI, Query
from model import model  # Импортируйте вашу модель

app = FastAPI()

@app.get("/", response_model=int)
async def predict(text: str = Query(...)):
    """
    Эндпоинт для предсказания. Принимает текстовый параметр в запросе.
    
    :param text: текст для предсказания
    :return: предсказание модели
    """
    # Получаем предсказание от модели
    output = model.predict(text)
    
    # Возвращаем результат
    return output

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)