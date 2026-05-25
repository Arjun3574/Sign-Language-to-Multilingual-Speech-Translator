import cv2
import mediapipe as mp
import numpy as np
import os
import json
import kagglehub
import glob
import tensorflow as tf
from tqdm import tqdm
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

print("✅ Connecting to Kaggle Dataset...")
dataset_path = kagglehub.dataset_download("prasadshet/indian-sign-language-video-dataset")
DATA_PATH = "Processed_Data_V2" 

# 1. THE SMART RADAR
print("🕵️‍♂️ Scanning deep inside the Kaggle folders for hidden MP4s...")
all_videos = glob.glob(os.path.join(dataset_path, '**', '*.mp4'), recursive=True)

class_videos = {}
for video_path in all_videos:
    class_name = os.path.basename(os.path.dirname(video_path))
    if class_name == "Sample Videos": continue
    if class_name not in class_videos: class_videos[class_name] = []
    class_videos[class_name].append(video_path)

print(f"🔥 FOUND IT! Discovered {len(class_videos)} real Sign Language words.")

# 2. MEDIA PIPE EXTRACTION
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

def extract_keypoints(detection_result):
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)
    if not detection_result.hand_landmarks: return np.concatenate([lh, rh])
    for idx, hand_handedness in enumerate(detection_result.handedness):
        hand_type = hand_handedness[0].category_name
        landmarks = detection_result.hand_landmarks[idx]
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        if hand_type == "Left": lh = coords
        elif hand_type == "Right": rh = coords
    return np.concatenate([lh, rh])

print("\n🚀 Local Extraction Starting (With Smart Resume)...")
all_tasks = []
for class_name, video_paths in class_videos.items():
    for video_file in video_paths:
        all_tasks.append((class_name, video_file))

for class_name, video_file in tqdm(all_tasks, desc="Extracting Videos", unit="video"):
    save_dir = os.path.join(DATA_PATH, class_name)
    os.makedirs(save_dir, exist_ok=True)
    
    filename = os.path.basename(video_file).replace('.mp4', '.npy')
    save_path = os.path.join(save_dir, filename)

    # ==========================================
    # 🚨 THE SMART RESUME UPGRADE 🚨
    # If the file already exists, instantly skip to the next one!
    if os.path.exists(save_path):
        continue
    # ==========================================
    
    cap = cv2.VideoCapture(video_file)
    frames_data = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        frames_data.append(extract_keypoints(detection_result))
    cap.release()

    if len(frames_data) > 0:
        np.save(save_path, np.array(frames_data))

print("✅ Extraction Complete! Formatting for AI...")

# 3. PREPARE DATA FOR 1D-CNN
SEQUENCE_LENGTH = 80
INPUT_SIZE = 126
dataset_features, dataset_labels = [], []
class_names = sorted(list(class_videos.keys()))
class_map = {name: i for i, name in enumerate(class_names)}

for class_name in class_names:
    class_dir = os.path.join(DATA_PATH, class_name)
    if not os.path.exists(class_dir): continue
    for file in os.listdir(class_dir):
        data = np.load(os.path.join(class_dir, file))
        if len(data) >= SEQUENCE_LENGTH:
            start = (len(data) - SEQUENCE_LENGTH) // 2
            data = data[start : start + SEQUENCE_LENGTH]
        else:
            padding = np.zeros((SEQUENCE_LENGTH - len(data), INPUT_SIZE))
            data = np.concatenate((data, padding))
        dataset_features.append(data)
        dataset_labels.append(class_map[class_name])

X = np.array(dataset_features)
y = to_categorical(dataset_labels, num_classes=len(class_names)).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. BUILD & TRAIN THE AI
print(f"🧠 Building Hybrid AI Model...")
model = Sequential([
    Input(shape=(SEQUENCE_LENGTH, INPUT_SIZE)),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    Bidirectional(GRU(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(GRU(32)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

print("🔥 Starting Real Training...")
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=1)

# 5. EXPORT YOUR FILES
print("💾 Saving Model...")
model.save('isl_hybrid_model.h5')
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)

print("🎉 DONE! Files are ready in your backend folder.")