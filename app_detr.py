# Import required libraries
import PIL
import numpy
import streamlit as st
import torch
import supervision as sv
from transformers import DetrForObjectDetection, DetrImageProcessor
import gdown
import sys

# Setting page layout
st.set_page_config(
    page_title="Non Scrap Material Detection",  # Setting page title
    page_icon="🤖",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded",    # Expanding sidebar by default
)

# Creating sidebar
with st.sidebar:
    st.header(":violet[Data Input Menu for your model]")     # Adding header to sidebar
    # Adding file uploader to sidebar for selecting images
    source_img = st.file_uploader(
        "Upload an image by clicking on the Browse button (or) you can drag and drop an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    # Model Options
    confidence = float(st.slider(
        "Select Model Confidence", 25, 100, 40)) / 100

# Creating main page heading
st.header(":rainbow[ISB AMPBA-B18 - Capstone Project]")
st.markdown("""<hr style="height:2px;border:none;color:#4B0082;background-color:#333;" /> """, unsafe_allow_html=True)
st.header("Non Recyclable Scrap Object Detection")
st.caption(':rainbow[Upload a image with Scrap material.]')
st.caption('Then click the :red[Detect Objects] button and check the result.')
# Creating two columns on the main page
col1, col2 = st.columns(2)

# Adding image to the first column if image is uploaded
with col1:
    if source_img:
        # Opening the uploaded image
        uploaded_image = PIL.Image.open(source_img)
        # Adding the uploaded image to the page with a caption
        st.image(source_img,
                 caption="Uploaded Image",
                 use_column_width=True
                 )
        
if st.sidebar.button('Detect Objects'):
    
    try:
 
        url = 'https://drive.google.com/drive/folders/1zz83NS1_QhALCr-3-DnGQrgGgsRF7ENo'
        gdown.download_folder(url)
    
        CHECKPOINT = 'custom-model'
    
        image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
        model = DetrForObjectDetection.from_pretrained(CHECKPOINT)
    except Exception as ex:
        st.error(
            f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

    with torch.no_grad():
        image = numpy.asarray(uploaded_image)
        inputs = image_processor(images=image, return_tensors='pt')
        outputs = model(**inputs)
        
        print("uploaded image", type(image))
        
        target_sizes = torch.tensor([image.shape[:2]])
        
        results = image_processor.post_process_object_detection(
            outputs=outputs,
            threshold=confidence,
            target_sizes=target_sizes
            )[0]
        
        detections = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=0.5)
        categories = {0: {'id': 0, 'name': 'Metal', 'supercategory': 'none'},
        1: {'id': 1, 'name': '-', 'supercategory': 'Metal'},
        2: {'id': 2, 'name': 'NonScrap', 'supercategory': 'Metal'}}

        id2label = {k: v['name'] for k,v in categories.items()}
        labels = [f"{id2label[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in detections]
        box_annotator = sv.BoxAnnotator()
        res = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

##########################################################

    with col2:
        st.image(res,
                 caption='Detected Image',
                 use_column_width=True
                 )