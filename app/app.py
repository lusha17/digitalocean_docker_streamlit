import queue
import urllib.request
from pathlib import Path
from typing import List, NamedTuple, Optional

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import (RTCConfiguration,WebRtcMode,WebRtcStreamerContext,webrtc_streamer,)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

CLASSES = [ 'bill', 'card', 'face', 'knife', 'mask', 'firearm', 'purse', 'smartphone']

@st.experimental_singleton
def generate_label_colors():
    return np.random.uniform(0, 255, size=(len(CLASSES), 3))

def detect(image, net):
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    return preds

def wrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []
    rows = output_data.shape[0]
    image_width, image_height, _ = input_image.shape
    x_factor = image_width / 640
    y_factor =  image_height / 640
    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:
            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):
                confidences.append(confidence)
                class_ids.append(class_id)
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 
    result_class_ids = []
    result_confidences = []
    result_boxes = []
    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])
    return result_class_ids, result_confidences, result_boxes

def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result

colors = generate_label_colors()

DEFAULT_CONFIDENCE_THRESHOLD = 0.5


class Detection(NamedTuple):
    name: str
    prob: float
        
cache_key = "object_detection_dnn"

if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = cv2.dnn.readNetFromONNX('all_s.onnx')
    st.session_state[cache_key] = net

#confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05)

def _annotate_image(image, detections):
    inputImage = format_yolov5(image)
    result: List[Detection] = []
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            name = CLASSES[idx]
            result.append(Detection(name=name, prob=float(confidence)))
            label = f"{name}: {round(confidence * 100, 2)}%"
            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image,label,(startX, y),cv2.FONT_HERSHEY_SIMPLEX,0.5,COLORS[idx],2,)
    return image, result

result_queue = (queue.Queue())

def callback(frame):
    image = frame.to_ndarray(format="bgr24")
    inputImage = format_yolov5(image)
    outs = detect(inputImage, net)
    class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])
    for (classid, confidence, box) in zip(class_ids, confidences, boxes):
        color = colors[int(classid) % len(colors)]
        image = cv2.rectangle(image, box, color, 2)
        image = cv2.rectangle(image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
        image = cv2.putText(image, CLASSES[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))
    #result_queue.put(result)  
    return av.VideoFrame.from_ndarray(image, format="bgr24")

webrtc_ctx = webrtc_streamer(key="object-detection",mode=WebRtcMode.SENDRECV,rtc_configuration=RTC_CONFIGURATION,video_frame_callback=callback,media_stream_constraints={"video": True, "audio": False},async_processing=True,)
if st.checkbox("Show the detected labels", value=False):
    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()
        while True:
            try:
                result = result_queue.get(timeout=1.0)
            except queue.Empty:
                result = None
            labels_placeholder.table(result)