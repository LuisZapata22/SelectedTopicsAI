# Sign Language Detection

## Project overview

This repository implements a real-time sign language recognition system built with deep learning and computer vision components. The project is organized into three stages:

1. **Model training** — initial training on a large dataset of hand-gesture images.
2. **Fine-tuning** — transfer learning and hyperparameter adjustments to improve accuracy and robustness.
3. **Deployment** — an interactive web application implemented with Streamlit for real-time webcam inference.

The solution combines a lightweight CNN backbone (MobileNetV3), MediaPipe for hand detection and localization, and a Streamlit front end for live prediction and visualization.

---

## Contents

* `signlanguagecnn.ipynb` — notebook with the initial model definition and training pipeline.
* `signlanguagecnn-ft-val.ipynb` — notebook used to fine-tune the trained model and validate performance.
* `stremlit_web.py` — Streamlit application that performs real-time inference using the webcam, MediaPipe, and the trained Keras model.
* `mobilenet_sign_language_model.keras` -  the final model keras file
* `.h5 files` - model weights including finetuned and not finetuned

---

## Design and architecture

### High-level flow

1. Capture webcam frames using OpenCV.
2. Detect and localize the hand using MediaPipe Hands.
3. Crop a square region around the detected hand with padding, resize to the model input size, and apply MobileNetV3 preprocessing.
4. Feed the processed crop to the Keras model for classification.
5. Overlay the predicted label and confidence on the video frame and display the result on the Streamlit page.

### Model

* **Backbone**: MobileNetV3 Large

* **Input size**: 200 × 200 × 3 (RGB).

* **Classes**: 29 classes covering A–Z and special tokens: `del`, `nothing`, `space`. The class list used by the app is:

  ```txt
  ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space']
  ```

* **Loss**: categorical crossentropy

* **Training strategy**: initial training on a large annotated dataset, followed by fine-tuning

---

## Streamlit application details (`stremlit_web.py`)

Key behaviors implemented:

* Loads a saved Keras model with `tf.keras.models.load_model()` wrapped by `st.cache_resource` to avoid re-loading on every interaction.
* Uses MediaPipe Hands for detection and tracking (`max_num_hands=1`, detection and tracking confidence thresholds are settable).
* Crops a square region around the detected hand with padding proportional to the bounding-box size to create the model input.
* Uses `tensorflow.keras.applications.mobilenet_v3.preprocess_input` for model preprocessing.
* Displays prediction text and the bounding box on the live video frame inside Streamlit.

Important parameters to consider in the script:

* `IMG_SIZE = (200, 200)` — expected input resolution for the model.
* `camera = cv2.VideoCapture(1)` — device index. Change to `0` if the default webcam is on index 0.
* Model file name expected by the app: `'mobilenet_sign_language_model.keras'`. Place your trained model artifact in the same working directory as `stremlit_web.py`, or modify the path accordingly.

---

## Installation and usage

### Recommended environment

* Python 3.8 — 3.11 (confirm compatibility with your TensorFlow version).
* GPU recommended for training (NVIDIA GPU + CUDA/cuDNN), not required for inference with small models.

### Example `requirements.txt`

A suggested list of packages used by the project:

```
tensorflow>=2.10
opencv-python
mediapipe
streamlit
numpy
pandas
matplotlib
scikit-learn
```

Adjust versions to match your environment and TensorFlow compatibility requirements.

### Run the Streamlit app

1. Install dependencies (recommended in a virtual environment):

```bash
python -m venv venv
source venv/bin/activate    # Linux / macOS
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

2. Place the trained model file (for example `mobilenet_sign_language_model.keras`) in the repository root or update the model path in `stremlit_web.py`.

3. Start the app:

```bash
streamlit run stremlit_web.py
```

4. If the camera fails to initialize, edit the camera index in `stremlit_web.py`:

```python
camera = cv2.VideoCapture(0)
```

or try other indices (0, 1, ...).

---


## Troubleshooting

* **Model not found**: ensure the filename set in `stremlit_web.py` points to an existing `.keras` model file.
* **Camera not opening**: try different camera indices (0, 1, …) or verify that no other process is using the device. On some systems, permission or driver issues can prevent access.
* **Slow inference**: consider converting the model to TensorFlow Lite or using a smaller backbone; ensure GPU drivers are properly configured if using GPU.
* **Incorrect crops / poor prediction**: verify MediaPipe hand detection output and ensure padding/crop logic produces centered hand images with minimal background.

---

## Extensibility and future work

* Add temporal modeling (RNN, LSTM, or transformers) to handle dynamic gestures and sequences.
* Expand the dataset to cover more signers, backgrounds, and lighting conditions to reduce domain shift.
* Provide a continuous text output (sentence building) with smoothing or a language model to map sequences of detected signs into words or phrases.
* Export the model to mobile-friendly formats (TFLite) and create mobile or embedded deployments.

---

## License

This project is provided under the MIT License. 

---

### Kaggle

The notebooks used in this project where trained in [kaggle](www.kaggle.com)
