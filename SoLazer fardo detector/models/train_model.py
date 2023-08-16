# Este arquivo pode conter a função para treinar seu modelo.

from preprocess_images import preprocess_images
from build_model import build_model

def train_model():
    datagen = preprocess_images()
    model = build_model()
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    train_gen = datagen.flow_from_directory(
        'data/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    
    val_gen = datagen.flow_from_directory(
        'data/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50
    )
    
    model.save('models/beverage_model.h5')
    
    return history

# ... outras funções relacionadas ao treinamento do modelo conforme necessário
