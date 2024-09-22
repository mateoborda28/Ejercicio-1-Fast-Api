from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import pandas as pd
import pickle
from typing import Optional
from sklearn.linear_model import Ridge
import numpy as np

# Crear una instancia de FastAPI
app = FastAPI()

# Definir el archivo JSON donde se guardarán las predicciones
file_name = 'predicciones.json'

# Cargar el modelo preentrenado desde el archivo pickle
model_path = "best_model.pkl"
with open(model_path, 'rb') as model_file:
    modelo = pickle.load(model_file)

# Cargar base de predicción en kaggle
prueba = pd.read_csv("prueba_APP.csv", sep = ";", header = 0, decimal = ",")

# Cargar funciones de la línea de código y listas
covariables = ['dominio', 'Tec', 'Avg. Session Length','Time on App',
                 'Time on Website', 'Length of Membership']


# Modelo de datos para la API (simplificado, adaptado según tus variables)
class InputData(BaseModel):
    email: str  # ID del usuario (no es covariable)
    dominio: str
    Tec: str
    Address: str
    Avg: float
    Time_App: float
    Time_Web: float
    Length: float

@app.get("/")
def home():
    # Retorna un simple mensaje de texto
    return 'Predicción Precio'

# Función para guardar predicciones en un archivo JSON
def save_prediction(prediction_data):
    try:
        with open(file_name, 'r') as file:
            predictions = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        predictions = []

    predictions.append(prediction_data)

    with open(file_name, 'w') as file:
        json.dump(predictions, file, indent=4)

# Endpoint para realizar la predicción
@app.post("/predict")
def predict(data: InputData):
    # Crear DataFrame a partir de los datos de entrada
    # Backend
    numericas = pd.DataFrame({'Avg. Session Length':[data.Avg],'Time on App':[data.Time_App],
    'Time on Website':[data.Time_Web], 'Length of Membership':[data.Length]})

    categoricas = pd.DataFrame({'dominio':[data.dominio], 'Tec':[data.Tec]})
  
    base_modelo = pd.concat([numericas, categoricas], axis=1)
    yhat = modelo.predict(base_modelo)

    base_modelo["y"] = prueba["price"].copy()


    ## alinearme al final
    prediction_label = np.round(float(yhat),2)

    # Guardar predicción con ID en el archivo JSON
    prediction_result = {"email": data.email, "prediction": prediction_label}
    save_prediction(prediction_result)

    return prediction_result

# Ejecutar la aplicación si se llama desde la terminal
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
