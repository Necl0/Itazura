from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pickle




# title of the app
st.title("Itazura")

# Specify canvas parameters in application
drawing_mode = "freedraw"
stroke_width = 3
realtime_update = True

# Create a canvas component
canvas_result = st_canvas(
    stroke_width=stroke_width,
    update_streamlit=realtime_update,
    height=600,
    width=600,
    background_color= "White",
    drawing_mode=drawing_mode,
    key="canvas",
    stroke_color="rgba(0, 0, 0, 1)",
)






if st.button("Predict"):
    #model = pickle.load(open('model.pkl', 'rb'))

    # convert canvas content to png
    img = Image.fromarray(np.uint8(canvas_result.image_data))


    img.save('temp.png')

    # convert image to numpy array
    img = 1 - (np.asarray(Image.open("./temp.png").convert("L").resize((224, 224))) / 255)

    # test
    print(np.sum(img))

    #prediction = model.predict(img[None, :, :]).argmax()

    #st.write(f"Prediction: {prediction}")
