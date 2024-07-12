import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model

model = load_model('model/Age_Sex_Detection.keras')

def predict_image(model, image):
    image = image.convert('RGB') if image.mode != 'RGB' else image
    image = np.expand_dims(np.array(image.resize((48, 48))) / 255.0, axis=0)
    predictions = model.predict(image)
    predicted_gender, predicted_age  = int(np.round(predictions[0][0, 0])), int(np.round(predictions[1][0, 0]))
    return predicted_gender, predicted_age

def on_click_predict(model, image):
    gender, age = predict_image(model, image)
    gender_str = "Male" if gender == 0 else "Female"
    st.write(f"Predicted Gender: {gender_str}")
    st.write(f"Predicted Age: {age}")

st.title("Age and Gender Prediction")
st.sidebar.title("Upload Image")

uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    upload_img = Image.open(uploaded_file)
    st.image(upload_img, caption='Uploaded Image', use_column_width=True)
    if st.button("Predict"): 
        on_click_predict(model, upload_img)


