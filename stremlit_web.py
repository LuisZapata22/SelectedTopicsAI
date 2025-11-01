import cv2
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import mediapipe as mp
import numpy as np

# --- Constants and Model Loading ---
IMG_SIZE = (200, 200)
CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Use st.cache_resource to load the model only once
@st.cache_resource
def get_model():
    model = load_model('mobilenet_sign_language_model.keras')
    return model

model = get_model()

# --- Preprocessing Function ---
def preprocess(img):
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, IMG_SIZE)
    
    if img.shape[-1] == 1:
        img = tf.image.grayscale_to_rgb(img)

    img_preprocessed = preprocess_input(img)
    return img_preprocessed

#  Prediction Function 
def make_prediction(_model, frame):
    frame = preprocess(frame)
    frame = tf.expand_dims(frame, axis=0)
    prediction = _model.predict(frame, verbose=0)
    return prediction

# Frame Processing Function 
def process_frame(frame, hands, _model):
    color = (59, 185, 247)
    
    # Flip image for mirror view and convert color
    frame = cv2.flip(frame, 1)
    # Convert BGR (OpenCV default) to RGB (MediaPipe/Keras convention)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Set image to immutable to improve performance
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    # Re-convert back to BGR for drawing with OpenCV
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Initialize prediction result
    prediction_text = "No Hand Detected"
    x_min, y_min = 40, 100

    if results.multi_hand_landmarks:
        h, w, _ = frame.shape
        
        # Use the first detected hand
        for hand_landmarks in results.multi_hand_landmarks:
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]

            # Calculate bounding box
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            # Calculate center and size
            box_size = max(x_max - x_min, y_max - y_min)
            cx = int((x_min + x_max) / 2)
            cy = int((y_min + y_max) / 2)

            # Determine square crop area with padding
            padding = int(box_size * 0.3)
            
            x1_crop = max(0, cx - box_size // 2 - padding)
            y1_crop = max(0, cy - box_size // 2 - padding)
            x2_crop = min(w, cx + box_size // 2 + padding)
            y2_crop = min(h, cy + box_size // 2 + padding)

            cropped_hand = frame[y1_crop:y2_crop, x1_crop:x2_crop]
            
            # Draw square around hand
            cv2.rectangle(frame, (x1_crop, y1_crop), (x2_crop, y2_crop), color, 2)
            
            if cropped_hand.size != 0:
                # Model prediction
                pred = make_prediction(_model, cropped_hand)
                predicted_index = np.argmax(pred[0])
                
                if predicted_index < len(CLASS_NAMES):
                    prediction_text = f"Prediction: {CLASS_NAMES[predicted_index]} ({pred[0][predicted_index]:.2f})"
                else:
                    prediction_text = "Prediction: Unknown Class"
                
                break 

    # Display the prediction on the main frame
    text_position = (int(x_min*0.7), int((y_min*0.7) - 10))
    cv2.putText(frame, prediction_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
    return frame


st.title("Sign Language Recognition")
run = st.checkbox('Run Camera')

FRAME_WINDOW = st.image([])

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

camera = cv2.VideoCapture(1) 

while run:
    success, frame = camera.read()
    if not success:
        st.error("Failed to read from camera. Check camera index (try 0 or 1).")
        break
    
    # Process the frame
    frame = process_frame(frame, hands, model)
    
    # Display the frame in Streamlit
    FRAME_WINDOW.image(frame, channels="BGR")

else:
    st.write('Camera stopped.')
    camera.release()

hands.close()