import numpy as np
from tensorflow.keras.models import load_model #type:ignore
from tensorflow.keras.preprocessing import image #type:ignore

model = load_model("cnn_model.h5")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        return "Real"
    else:
        return "Fake"