
# ✋ MULTI GESTURE DETECTOR 🤖

This project is a real-time hand gesture recognition system using **MediaPipe**, **OpenCV**, and a **TensorFlow/Keras deep learning model**. It allows users to perform various hand gestures that are detected live via webcam.

---

## 🚀 Features

- ✅ Real-time hand gesture detection using webcam
- 🧠 Custom-trained model using TensorFlow/Keras
- 📷 MediaPipe for accurate hand landmark tracking
- 🎯 Confidence threshold and cooldown logic for stable predictions
- 🔁 Support for multiple gestures including:
  - Hello 👋
  - Fist ✊
  - Thumbs Up 👍
  - Thumbs Down 👎
  - Peace ✌️
  - Stop ✋
  - Okay 👌
  - Rock 🤘
  - Call Me 🤙

---

## 🛠️ Tech Stack

- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras
- NumPy

---

## 📁 Directory Structure

```
MULTI-GESTURE-DETECTOR/
│
├── model.keras                 # Trained gesture classification model
├── gestures/                  # Folder containing .csv data for each gesture
├── gestures.txt               # List of gesture labels
├── collect_data.py            # Script to record gesture data
├── train_model.py             # Script to train the model
├── predict_gesture.py         # Main script for real-time prediction
└── README.md                  # This file
```

---

## 🧪 How to Run

1. **Install Dependencies**  
   ```bash
   pip install opencv-python mediapipe numpy tensorflow
   ```

2. **Run Real-Time Prediction**  
   ```bash
   python predict_gesture.py
   ```

3. **Press `q` to quit the live window**

---

## 🧠 Model Training

To train your own model:

1. Record data using `collect_data.py`
2. Train the model using `train_model.py`
3. Ensure the model is saved as `model.keras`

---

## 🤩 Future Enhancements

- 🔥 Two-hand gesture detection
- 🌐 Web-based UI with Flask
- 💾 Save gesture history/log
- 🧠 Integrate voice feedback or system control

---

## 📜 License

This project is licensed under the MIT License.

---

## 👤 Author

**Vansh Agrawal**  
[GitHub](https://github.com/vansh070605) | [LinkedIn](https://www.linkedin.com/in/vansh070605)
