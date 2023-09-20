import streamlit as st
import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

# Inicializa o PaddleOCR
ocr = PaddleOCR()


def draw_boxes_on_image_nlp(image, results):
    img_with_boxes = image.copy()
    for result in results[0]:
        text = result[1][0]
        points = result[0]

        points = [(int(p[0]), int(p[1])) for p in points]

        cv2.putText(img_with_boxes, text, points[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return img_with_boxes


def main():
    st.title("Aplicação do OCR com Streamlit")
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image('img/06.png', use_column_width=True)
    with col2:
        st.title("NLP")
    uploaded_image = st.file_uploader("Escolha uma imagem...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        with st.spinner('Detectando texto...'):
            image = Image.open(uploaded_image)
            image = np.array(image)  # converte a imagem PIL para uma matriz NumPy

            # Cache da função para economizar tempo de processamento
            @st.cache_resource
            def process_image(image):
                results = ocr.ocr(image)
                return results

            results = process_image(image)

            image_with_boxes = draw_boxes_on_image_nlp(image, results)

            st.image(image_with_boxes, caption='Imagem com Caixas Delimitadoras', use_column_width=True)


if __name__ == "__main__":
    main()