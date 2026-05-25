# 🤟 SignBridge: The Next-Generation Indian Sign Language Translator

**SignBridge** is a bleeding-edge, full-stack Artificial Intelligence application that bridges the communication gap for the deaf and hard-of-hearing communities. By leveraging deep learning, temporal tracking neural networks, and modern Neural Machine Translation (NMT), SignBridge translates continuous Indian Sign Language (ISL) gestures into real-time multilingual text and highly natural text-to-speech audio.

Whether you are signing live through a webcam or uploading a pre-recorded video, SignBridge intelligently tracks your hand coordinates and reconstructs the meaning using robust context-aware AI.

---

## 🌟 Core Features

- 🎥 **Real-Time WebCam Streaming**: Captures live browser video at 30+ FPS, communicating with the backend cleanly via WebSockets/REST with micro-second thread-lock handling for uninterrupted AI detection.
- 📁 **Offline Video Upload & Parsing**: A dedicated offline processing engine seamlessly reads uploaded `.mp4` video files. It rigorously extracts center-cropped 80-frame dimensional matrices to exactly mirror standard Kaggle ML training data for unmatched prediction accuracy.
- 🧠 **Temporal Deep Learning**: Unlike basic classifiers, SignBridge utilizes a **Bi-GRU (Bidirectional Gated Recurrent Unit)** capable of understanding the *motion* of signs over time (up to 80-frame contexts at once).
- 🌍 **Meta NLLB-200 Integration (No API Limits)**: Powered by Facebook/Meta's local NLLB 600M parameter model, it seamlessly translates detected English signs into heavily localized regional languages (Hindi, Tamil, Marathi, Bengali, Spanish, French, etc.) instantly.
- 🔊 **Regional Text-to-Speech (Google TTS)**: Generates high-fidelity `.mp3` audio dynamically on the python backend securely bypassing strict browser limitations for foreign languages.

---

## 🏗 How It Works (Project Architecture)

The project elegantly handles complex, multi-layered data pipelines locally on your machine without relying on paid APIs:

1. **Step 1: Data Capture (Frontend)**
   Beautifully designed Glassmorphism vanilla-HTML/JS UI accesses the hardware web-camera tracking the user's upper body. 
2. **Step 2: Skeletal Extraction (Computer Vision)**
   The backend routes frames immediately to **Google MediaPipe HandLandmarker**. The system captures the exact 3D Cartesian coordinates (`x, y, z`) of 42 individual hand joints (21 per hand).
3. **Step 3: Temporal Motion Buffer (Data Mapping)**
   For live prediction, continuous geometry arrays are mathematically buffered exactly 80 frames deep. Blank frames correctly push `zero-tensors` ensuring time doesn't warp if a hand is dropped.
4. **Step 4: AI Execution (Machine Learning)**
   The 80x126 numerical matrix passes through the loaded **TensorFlow/Keras Hybrid Model**, which outputs the highest confidence word mathematically describing the sequence.
5. **Step 5: Natural Language Pipeline (NLP)**
   The generated text is submitted through `transformers` into the heavy `facebook/nllb-200-distilled-600M` model. By aggressively converting string tokens to target semantic `<eos>` IDs natively, the context translates securely (e.g., "Hello world" -> "नमस्ते दुनिया"). 
6. **Step 6: Voice Synthesis**
   Using `gTTS`, the Python layer converts the Devanagari regional scripts into audible, playable `.mp3` format which is instantly routed back to the DOM Audio interface.

---

## 💻 Full Technology Stack

### **Frontend / Client**
- **HTML5 & CSS3**: Custom modern aesthetics, dark-mode styling, scanning ray-cast animations.
- **Vanilla JavaScript**: Dynamic element targeting, native HTML WebRTC camera routing, Base64 JSON packaging.

### **Backend / Server**
- **Python (3.x)** & **Flask**: Multi-threaded lightweight inference handling.
- **OpenCV (`cv2`)**: Real-time matrix conversion, scaling, cropping, and mp4 video reader handlers.
- **Google TTS**: Python-native automated audio synthesis.

### **Machine Learning Models**
- **Google MediaPipe**: Fast C++ powered spatial coordinate tracker.
- **TensorFlow & Keras**: The core neural inference execution engine relying on a massive 1D-CNN and Bidirectional Recurrent Layer.
- **Hugging Face (`transformers`, `sentencepiece`, `torch`)**: Hugging face library housing the 2.5 GB heavy Seq-2-Seq translation neural net.

---

## 🚀 Getting Started

To run the full SignBridge web stack natively on your PC:

### 1. Install System Requirements
Ensure Python is installed, then download the neural dependencies:
```bash
pip install -r requirements.txt
```
*(Note: Initial runs will dynamically download the ~2.5 GB NLLB-200 model locally into your cache).*

### 2. Boot the Server
Run the Flask architecture on your local port:
```bash
python app.py
```

### 3. Start Translating
Open your favorite internet browser and surf exactly to:
**http://127.0.0.1:5000**
Allow your Web Camera access, configure your preferred final output language, and begin signing!