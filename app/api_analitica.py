from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    # Cargar modelo previamente entrenado
    print("Intentando cargar el modelo...")
    model = joblib.load(r"D:\SEMESTRE 2025-I\ANALÍTICA DE NEGOCIOS\modelo_demencia_actualizado.pkl")
    print("Modelo cargado exitosamente")
except Exception as e:
    print(f"Error al cargar el modelo: {str(e)}")
    exit(1)

class PredictionInput(BaseModel):
    age: float
    gender: float
    educationyears: float
    Global: float
    EF: float
    PS: float
    glucose_min: float
    cholesterol_total: float
    hypertension_sys: float
    smoking: float
    Fazekas: float
    lacunes_num: float
    SVD_Simple_Score: float
    CMB_count: float

    # Configurar alias para manejar nombres con espacios
    model_config = {
        "alias_generator": lambda x: x.replace('_', ' '),
        "populate_by_name": True
    }

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        # Verificar que todos los campos sean numéricos
        for field, value in input_data.model_dump().items():
            if not isinstance(value, (int, float)):
                raise HTTPException(
                    status_code=400,
                    detail=f"El campo '{field}' debe ser un número válido"
                )

        # Convertir input a arreglo
        data = np.array([
            input_data.age,
            input_data.gender,
            input_data.educationyears,
            input_data.Global,
            input_data.EF,
            input_data.PS,
            input_data.glucose_min,
            input_data.cholesterol_total,
            input_data.hypertension_sys,
            input_data.smoking,
            input_data.Fazekas,
            input_data.lacunes_num,
            input_data.SVD_Simple_Score,
            input_data.CMB_count
        ]).reshape(1, -1)

        # Realizar predicción
        prob = model.predict_proba(data)[0][1]
        pred = int(prob >= 0.5)

        # Interpretar riesgo
        if prob >= 0.75:
            riesgo = "Alto"
        elif prob >= 0.5:
            riesgo = "Moderado"
        else:
            riesgo = "Bajo"

        return {
            'probabilidad_demencia': round(prob, 4),
            'riesgo': riesgo,
            'prediccion': pred,
            'status': 'success'
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Error en los datos de entrada: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    print("Iniciando API de predicción con FastAPI...")
    print("API disponible en: http://localhost:8000")
    print("Documentación: http://localhost:8000/docs")
    uvicorn.run("api_analitica:app", host="0.0.0.0", port=8000, reload=True)
