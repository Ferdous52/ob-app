import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import onnxruntime as ort
import numpy as np
import cv2

# Load ONNX model
@st.cache_resource
def load_model():
    session = ort.InferenceSession("yolov11n.onnx")  # Replace with your ONNX model path
    return session

# Preprocess frame for YOLO ONNX
def preprocess(image):
    img = cv2.resize(image, (640, 640))
    img = img / 255.0
    img = img.transpose(2, 0, 1).astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

# Postprocess YOLO output (basic box filtering)
def postprocess(outputs, threshold=0.5):
    predictions = outputs[0][0]
    boxes = []
    for pred in predictions:
        x1, y1, x2, y2, conf = pred[:5]
        if conf > threshold:
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes

# Video transformer class for webcam
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.session = load_model()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        input_tensor = preprocess(img)
        outputs = self.session.run(None, {"images": input_tensor})
        boxes = postprocess(outputs)

        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return img

# Streamlit UI
st.title("📹 YOLO ONNX Real-Time Detection")
st.write("Accessing webcam... Please allow camera access.")

webrtc_streamer(
    key="yolo-onnx",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
