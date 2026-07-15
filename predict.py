import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE=(224,224)
model=tf.keras.models.load_model("brain_tumor_model.keras")

with open("class_names.json","r") as f:
    class_names= json.load(f)

def predict_image(image):
    image=image.convert("RGB")
    image=image.resize(IMG_SIZE)

    img_array=np.array(image)
    img_array=np.expand_dims(img_array,axis=0)
    img_array=preprocess_input(img_array)
    
    prediction = model.predict(img_array)[0]
    
    predicted_index=int(np.argmax(prediction))
    predicted_class=class_names[predicted_index]
    confidence=float(prediction[predicted_index])

    return predicted_class, confidence, prediction