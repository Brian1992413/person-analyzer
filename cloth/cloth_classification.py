import tensorflow as tf
import numpy as np
from PIL import Image

def detect_type_clothes(img_path):
    image = Image.open(img_path)
    image = image.resize((28, 28))
    image = image.convert("L")
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = (np.expand_dims(image_array,0))
    
    loaded_model = tf.keras.models.load_model('cloth/model.h5')
    probability_model = tf.keras.Sequential([loaded_model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(image_array)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return class_names[np.argmax(predictions)]