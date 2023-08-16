# Este arquivo pode conter funções para pré-processar suas imagens,
# como redimensionamento, normalização, aumento de dados, etc.

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_images():
    datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        validation_split=0.2
    )
    return datagen

# ... outras funções de pré-processamento conforme necessário
