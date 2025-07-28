import numpy as np
import PIL.Image as Image
import tensorflow as tf
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from warnings import filterwarnings
filterwarnings('ignore')

def streamlit_config():
    st.set_page_config(page_title='Classification', layout='centered')
    
    base64_image = """
    <style>
    body, .stApp {
        background-color: #006400 !important;
    }
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0) !important;
    }
    .bottom-right {
        position: fixed;
        bottom: 10px;
        right: 10px;
        background-color: rgba(0, 0, 0, 0.7);
        padding: 10px;
        border-radius: 10px;
        color: white;
        font-size: 16px;
    }
    </style>
    """
    st.markdown(base64_image, unsafe_allow_html=True)
    
    st.markdown(f'<h1 style="text-align: center; color: white;">Potato Disease Finder </h1>', unsafe_allow_html=True)
    add_vertical_space(4)
    
    # Show "by Varun Tiwari" at the start page
    st.markdown('<div class="bottom-right">by Varun Tiwari</div>', unsafe_allow_html=True)

streamlit_config()

def prediction(image_path, class_names=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']):
    img = Image.open(image_path)
    img_resized = img.resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    model = tf.keras.models.load_model('model.h5')
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(np.max(prediction) * 100, 2)

    st.markdown(
        f'<div class="bottom-right">Predicted Class: {predicted_class}<br>Confidence: {confidence}%</div>', 
        unsafe_allow_html=True
    )
    st.image(img.resize((400, 300)))

col1, col2, col3 = st.columns([0.1, 0.9, 0.1])
with col2:
    input_image = st.file_uploader(label='Upload the Image', type=['jpg', 'jpeg', 'png'])

if input_image is not None:
    col1, col2, col3 = st.columns([0.2, 0.8, 0.2])
    with col2:
        prediction(input_image)

