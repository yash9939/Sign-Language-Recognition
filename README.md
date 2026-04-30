## ✋ Sign Language Recognition with Gesture-to-Text Translation

<p align="center">
  <b>Bridging communication gaps using AI 🤖</b><br>
  <i>Computer Vision + Deep Learning + NLP</i>
</p>

---

## 🌟 Overview

This project presents a **real-time Sign Language Recognition system** that converts hand gestures into text using a webcam.

It combines:

* 📷 Computer Vision (MediaPipe)
* 🧠 Deep Learning (PyTorch)
* 🤖 NLP (N-gram model)

👉 **End-to-End Pipeline**

```text
Webcam → Hand Detection → Landmarks → Neural Network → Letter → Text → NLP → Output
```

---

## 🎯 Key Features

✨ Real-time hand gesture detection
🔤 Alphabet recognition (A–Z)
🧠 Deep Learning-based classification
✍️ Live text generation
🤖 NLP-based word suggestions
⚡ Auto-complete functionality
📊 Confidence score + progress bar
🎨 Modern UI with overlay & animations

---

## 🛠️ Tech Stack

| Category        | Tools           |
| --------------- | --------------- |
| Programming     | Python 🐍       |
| Computer Vision | OpenCV 📷       |
| Hand Tracking   | MediaPipe ✋     |
| Deep Learning   | PyTorch 🧠      |
| NLP             | N-gram Model 🤖 |

---

## 📂 Project Structure

```bash
Sign-Language-Recognition/
│
├── landmark_app.py              # 🚀 Main application
├── train_landmark_model.py      # 🧠 Model training
├── landmark_data_collection.py  # 📊 Dataset collection
├── ngram_model.py               # 🤖 NLP model
├── corpus.txt                   # 📚 Word corpus
├── landmark_model.pth           # 💾 Trained model
├── landmark_dataset/            # 📁 Dataset (CSV)
├── requirements.txt             # 📦 Dependencies
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/sign-language-recognition.git
cd sign-language-recognition
```

### 2️⃣ Create Virtual Environment (Optional)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python landmark_app.py
```

---

## 🎮 Controls

| Key       | Action                |
| --------- | --------------------- |
| ␣ (Space) | Add space             |
| D         | Delete last character |
| C         | Clear text            |
| 1         | Auto-complete word    |
| Q         | Quit                  |

---

## 🧠 How It Works

### 🔹 Step 1: Hand Detection

MediaPipe detects **21 landmarks** of the hand.

### 🔹 Step 2: Feature Extraction

* Each landmark → (x, y)
* Total features = **42**

### 🔹 Step 3: Deep Learning Model

* Input → 42 features
* Output → Alphabet (A–Z)

### 🔹 Step 4: Smoothing

* Uses buffer (deque)
* Eliminates flickering predictions

### 🔹 Step 5: NLP Prediction

* Uses **Bigram Model**
* Suggests words from partial input

👉 Example:

```text
Input: HE → Output: HELLO, HELP
```

---

## 🧠 Model Architecture

```text
Input: 42 features

→ Dense (256) + BatchNorm + ReLU + Dropout  
→ Dense (128) + ReLU + Dropout  
→ Dense (64) + ReLU  
→ Output (26 classes)
```

---

## 📊 Performance

* ✅ Accuracy: ~85%
* ⚡ Real-time processing
* 👍 Works in standard lighting

---

## ⚠️ Limitations

* Similar gestures (M/N, U/V) may confuse model
* Performance affected by lighting
* Limited dataset size

---

## 🔮 Future Improvements

🚀 CNN-based image model
🔊 Text-to-Speech output
📱 Mobile/Web deployment
🧠 Grammar correction (advanced NLP)
📊 Larger dataset for better accuracy

---

## 📸 Demo

<p align="center">
  <i>Add screenshots or demo GIF here</i>
</p>

---

## 👨‍💻 Authors

* **Shivam Kumar**
* **Yash Bharadwaj**

---

## 🏆 Resume Highlight

> Developed a real-time ASL recognition system using deep learning and NLP-based word prediction achieving ~85% accuracy.

---

## 📚 References

* MediaPipe Documentation
* PyTorch Documentation
* OpenCV Documentation
* N-gram Language Models

---

## ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!
