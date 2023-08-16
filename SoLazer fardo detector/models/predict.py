# Este arquivo pode conter a função para fazer previsões em novas imagens.

from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

def predict(image_path):
    model = load_model('models/beverage_model.h5')
    
    image = Image.open(image_path)
    image = image.resize((224, 224))
    
    image_array = np.expand_dims(np.array(image), axis=0) / 255.0
    
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    
    return predicted_class

# ... outras funções relacionadas à previsão conforme necessário
