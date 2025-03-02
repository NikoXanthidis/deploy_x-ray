from fastapi import FastAPI, File, UploadFile  # Importa FastAPI e UploadFile para manipular uploads / Imports FastAPI and UploadFile to handle uploads
from fastapi.responses import JSONResponse  # Importa JSONResponse para retornar respostas em JSON / Imports JSONResponse to return JSON responses
import tensorflow as tf  # Importa TensorFlow para carregar o modelo / Imports TensorFlow to load the model
import numpy as np  # Importa NumPy para operações numéricas / Imports NumPy for numerical operations
from PIL import Image  # Importa PIL para manipulação de imagens / Imports PIL for image handling
import io  # Importa io para lidar com arquivos em memória / Imports io to handle in-memory files
from fastapi.middleware.cors import CORSMiddleware  # Middleware para permitir acesso do frontend / Middleware to allow frontend access

# Inicializa a API FastAPI / Initializes the FastAPI app
app = FastAPI()

# Habilita o CORS para permitir que o frontend acesse a API / Enables CORS to allow frontend access to the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite solicitações de qualquer origem / Allows requests from any origin
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos (GET, POST, etc.) / Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos os cabeçalhos / Allows all headers
)

# Carrega o modelo treinado do arquivo "weights.keras" / Loads the trained model from the "weights.keras" file
model = tf.keras.models.load_model("weights.keras")  # Certifique-se de que o arquivo está no diretório correto / Ensure the file is in the correct directory

# Define as classes que o modelo pode prever / Defines the classes that the model can predict
class_names = ["Covid-19", "Normal", "Pneunomia viral", "Pneunomia bacterial"]  

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Pré-processa a imagem para o modelo / Preprocesses the image for the model"""
    image = image.resize((256, 256))  # Redimensiona a imagem para o tamanho esperado pelo modelo / Resizes the image to the model's expected size
    image = np.array(image) / 255.0   # Normaliza os valores dos pixels entre 0 e 1 / Normalizes pixel values between 0 and 1
    image = np.expand_dims(image, axis=0)  # Adiciona uma dimensão extra para representar um batch / Adds an extra dimension to represent a batch
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Recebe uma imagem, processa e retorna a classe predita / Receives an image, processes it, and returns the predicted class"""
    try:
        image_bytes = await file.read()  # Lê a imagem enviada pelo usuário / Reads the image sent by the user
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Converte a imagem para RGB / Converts the image to RGB
        processed_image = preprocess_image(image)  # Pré-processa a imagem antes da previsão / Preprocesses the image before prediction
        
        predictions = model.predict(processed_image)  # Faz a previsão usando o modelo carregado / Makes the prediction using the loaded model
        predicted_class = class_names[np.argmax(predictions)]  # Obtém a classe com maior probabilidade / Gets the class with the highest probability

        return JSONResponse(content={"class": predicted_class, "confidence": float(np.max(predictions))})  # Retorna a previsão e a confiança / Returns the prediction and confidence
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)  # Retorna erro em caso de falha / Returns an error in case of failure
