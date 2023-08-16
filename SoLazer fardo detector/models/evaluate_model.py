# Este arquivo pode conter a função para avaliar seu modelo treinado.

from tensorflow.keras.models import load_model
from preprocess_images import preprocess_images

def evaluate_model():
    datagen = preprocess_images()
    
    val_gen = datagen.flow_from_directory(
        'data/test',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    
    model = load_model('models/beverage_model.h5')
    
    loss, accuracy = model.evaluate(val_gen)
    print('Test Accuracy:', accuracy)
    
    return loss, accuracy

# ... outras funções relacionadas à avaliação do modelo conforme necessário
