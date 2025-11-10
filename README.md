# Sign Language Recognition Project

This project implements a real-time American Sign Language (ASL) alphabet recognition system using **MobileNetV3** and **MediaPipe** for hand detection and segmentation. The goal is to provide an accessible tool capable of classifying static hand gestures corresponding to the ASL alphabet.

---

## Overview

The solution is structured into two main stages:

1. **Model Training (MobileNetV3 - Transfer Learning & Fine-Tuning):**

   * A MobileNetV3-Large model pretrained on ImageNet was used as the base.
   * Data augmentation and fine-tuning were applied to improve generalization.
   * The model outputs predictions across **29 classes** (A–Z + `del`, `nothing`, and `space`).

2. **Real-Time Inference Web App (Streamlit + MediaPipe):**

   * MediaPipe identifies and segments the hand in each webcam frame.
   * The cropped hand image is resized and fed into the trained MobileNetV3 model.
   * Predictions are displayed live on the screen.

---

## Dataset

The model is trained on the **ASL Alphabet Dataset**, containing approximately **87,000 images** across **29 classes**. Each image is **200x200 pixels**. Data augmentation techniques used in training include:

* Random Rotation
* Random Cutout
* Random Zoom

These transformations increase robustness to variations in lighting, orientation, and hand appearance.

---

## Model Details

* **Architecture:** MobileNetV3-Large (Transfer Learning)
* **Input Size:** 200 × 200 × 3 (RGB)
* **Output Classes:** 29 (A–Z, del, space, nothing)
* **Loss Function:** Categorical Crossentropy
* **Optimization:** Adam
* **Fine-Tuning:** Selective layer unfreezing for improved feature adaptation

The final fine-tuned model achieves **high accuracy and strong generalization** abilities across validation datasets.

---

## Real-Time Application (Streamlit)

The Streamlit application:

1. Captures frames through the user's webcam.
2. Uses **MediaPipe Hands** to detect and isolate the hand region.
3. Crops and preprocesses the hand image.
4. Runs inference using the trained MobileNetV3 model.
5. Displays the predicted class and confidence score.

### Running the App

```bash
pip install -r requirements.txt
streamlit run stremlit_web.py
```

If the webcam doesn't load, adjust the camera index in the script:

```python
camera = cv2.VideoCapture(0)  # Try 1, 2, etc. if necessary
```

---

## Requirements

* Python 3.8+
* TensorFlow
* OpenCV
* MediaPipe
* Streamlit

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Vision Transformer (ViT) Model

In addition to the MobileNetV3-based approach, a **Vision Transformer (ViT-Base)** model was developed and evaluated for comparison. The ViT model treats images as sequences of fixed-size patches and applies self-attention to learn spatial relationships. This offers an alternative to convolution-based feature extraction.

### Key Details

* **Base Model:** ViT-Base (google/vit-base-patch16-224)
* **Framework:** PyTorch + Hugging Face Transformers
* **Input Size:** 224 × 224 × 3
* **Training:** Fine-tuning applied with a balanced dataset of approximately **20,000 images**
* **Output Classes:** Same 29 ASL alphabet classes

### Performance

The ViT model achieved:

* **Training Loss:** ~0.038 after only 4 epochs
* **Test Set Accuracy:** ~82% on a larger mixed dataset

While the ViT demonstrated **fast convergence** and **strong generalization**, the MobileNetV3 model remained better suited for **real-time inference**, particularly due to:

* Lower computational cost
* Faster inference speeds
* Compatibility with lightweight deployment

Therefore, **MobileNetV3 was selected for the real-time Streamlit application**, while **ViT is retained as a research benchmark** for future development.

## Limitations and Future Work

* Expansion to **dynamic gestures** (multi-frame sequence recognition)
* Integration of a **language model** for full sentence reconstruction
* Deployment to **mobile devices** using TensorFlow Lite
* Improved lighting and background robustness via additional preprocessing

---

## Contributors
* [Luis Fernando Zapata Moya](https://github.com/LuisZapata22)
* [Leonardo Nicolas Ampuero Terceros](https://github.com/VRL-458)



