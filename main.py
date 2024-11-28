from fastapi import FastAPI, Query
from scr.model import Model
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Model()
model.load_state_dict(torch.load('models/model_epoch_50.pt', map_location=device))
model.eval()

app = FastAPI()

@app.get("/", response_model=float)
async def predict(text: str = Query(...)):
    """
    Эндпоинт для предсказания. Принимает текстовый параметр в запросе.
    
    :param text: текст для предсказания
    :return: предсказание модели
    """
    # Получаем предсказание от модели
    output = round(float(model.predict(text).item()), 5)
    
    # Возвращаем результат
    return output

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)