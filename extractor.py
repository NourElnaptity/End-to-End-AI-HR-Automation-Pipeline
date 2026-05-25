import easyocr
import pypdf
import fitz  # PyMuPDF
import cv2
from PIL import Image
import numpy as np
import streamlit as st

@st.cache_resource
def load_ocr_model():
    return easyocr.Reader(['ar', 'en'], gpu=False)

def extract_text_and_boxes_from_image(image, reader):
    image_np = np.array(image)
    if len(image_np.shape) == 3 and image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    results = reader.readtext(image_np)
    text = " ".join([res[1] for res in results])
    
    # Draw boxes
    for (bbox, t, prob) in results:
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        br = (int(br[0]), int(br[1]))
        cv2.rectangle(image_np, tl, br, (0, 255, 0), 2)
        
    return text, Image.fromarray(image_np)

def extract_text_from_pdf(pdf_file, reader):
    reader_pdf = pypdf.PdfReader(pdf_file)
    text = ""
    for page in reader_pdf.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
            
    images_with_boxes = []
    
    if len(text.strip()) < 50:
        # Scanned PDF logic
        st.info("PDF appears to be scanned. Extracting using OCR...")
        pdf_file.seek(0)
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_text, img_boxed = extract_text_and_boxes_from_image(img, reader)
            text += page_text + "\n"
            images_with_boxes.append(img_boxed)
            
    return text, images_with_boxes
