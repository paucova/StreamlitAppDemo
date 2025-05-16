import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions


# Pre trained model
model = load_model("my_model.keras")
class_names = ['Avión', 'Automóvil', 'Pájaro', 'Gato', 'Ciervo',
               'Perro', 'Rana', 'Caballo', 'Barco', 'Camión']

# Ui - i got this from internet
st.title("Image Classification with Saul's model")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    img = img.resize((32, 32))
    image_array = image.img_to_array(img)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    prediction = model.predict(image_array)

    top_indices = prediction[0].argsort()[-3:][::-1]
    decoded_predictions = [(class_names[i], float(prediction[0][i])) for i in top_indices]
    
    st.subheader("Top Predictions:")
    for label, score in decoded_predictions:
        st.write(f"**{label}**: {score*100:.2f}%")
