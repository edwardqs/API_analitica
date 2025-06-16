from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import os
from pydantic import BaseModel, Field

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Intentar cargar el modelo al iniciar
try:
    print("Intentando cargar el modelo...")
    ruta_modelo = os.path.join(os.path.dirname(__file__), "modelo", "modelo_demencia_actualizado.pkl")
    model = joblib.load(ruta_modelo)
    print("Modelo cargado exitosamente")
except Exception as e:
    print(f"Error al cargar el modelo: {str(e)}")
    raise RuntimeError("No se pudo cargar el modelo")

# Entrada del modelo
class PredictionInput(BaseModel):
    age: float = Field(..., alias="age")
    gender: float = Field(..., alias="gender")
    educationyears: float = Field(..., alias="educationyears")
    Global: float = Field(..., alias="Global")
    EF: float = Field(..., alias="EF")
    PS: float = Field(..., alias="PS")
    glucose_min: float = Field(..., alias="glucose_min")
    cholesterol_total: float = Field(..., alias="cholesterol_total")
    hypertension_sys: float = Field(..., alias="hypertension_sys")
    smoking: float = Field(..., alias="smoking")
    Fazekas: float = Field(..., alias="Fazekas")
    lacunes_num: float = Field(..., alias="lacunes_num")
    SVD_Simple_Score: float = Field(..., alias="SVD_Simple_Score")
    CMB_count: float = Field(..., alias="CMB_count")

    class Config:
        allow_population_by_field_name = True

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
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

        # Predecir
        prob = model.predict_proba(data)[0][1]
        pred = int(prob >= 0.5)

        # Clasificación de riesgo
        riesgo = "Bajo"
        if prob >= 0.75:
            riesgo = "Alto"
        elif prob >= 0.5:
            riesgo = "Moderado"

        return {
            "probabilidad_demencia": round(prob, 4),
            "riesgo": riesgo,
            "prediccion": pred,
            "status": "success"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Error en los datos de entrada: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

# No usar if __name__ == '__main__' en producción FastAPI (esto se lanza con uvicorn)
if __name__=="main":
    import uvicorn
    port= int(os.getnev("PORT", 8000)
    univcorn.run(app, host="0.0.0.0", port=port)
