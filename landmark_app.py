import cv2
import torch
import mediapipe as mp
import torch.nn as nn
from ngram_model import NGramModel
import torch.nn.functional as F
from collections import deque

import time

# =========================
# CLASSES
# =========================
CLASSES = sorted(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

# =========================
# MODEL
# =========================
class LandmarkModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(42, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 26)
        )

    def forward(self, x):
        return self.net(x)

# =========================
# LOAD MODEL
# =========================
model = LandmarkModel()
model.load_state_dict(torch.load("landmark_model.pth", map_location=torch.device("cpu")))
model.eval()

# =========================
# LOAD N-GRAM MODEL
# =========================
ngram = NGramModel(n=2)

with open("corpus.txt", "r") as f:
    corpus_text = f.read()

ngram.train(corpus_text)

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

# =========================
# SMOOTHING + TEXT SYSTEM
# =========================
buffer = deque(maxlen=15)   # bigger buffer = more stability
stable_label = ""

sentence = ""
last_added = ""
hold_start = 0
hold_time = 1.5   # seconds to confirm letter

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    label = ""
    conf_val = 0.0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_list, y_list = [], []

            for lm in hand_landmarks.landmark:
                x_list.append(lm.x)
                y_list.append(lm.y)

            # NORMALIZATION
            base_x = x_list[0]
            base_y = y_list[0]

            x_list = [x - base_x for x in x_list]
            y_list = [y - base_y for y in y_list]

            max_val = max(
                max(abs(x) for x in x_list),
                max(abs(y) for y in y_list)
            )

            if max_val == 0:
                continue

            data = []
            for x, y in zip(x_list, y_list):
                data.append(x / max_val)
                data.append(y / max_val)

            data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

            # PREDICTION
            with torch.no_grad():
                output = model(data)
                prob = F.softmax(output, dim=1)
                confidence, pred = torch.max(prob, 1)

            conf_val = confidence.item()

            # LOWER threshold slightly
            if conf_val > 0.35:
                label = CLASSES[pred.item()]
            else:
                label = ""

    # =========================
    # SMOOTHING
    # =========================
    buffer.append(label)

    if len(buffer) == buffer.maxlen:
        most_common = max(set(buffer), key=buffer.count)

        if most_common != "":
            if stable_label != most_common:
                stable_label = most_common
                hold_start = time.time()

            # HOLD LOGIC → add letter
            elif time.time() - hold_start > hold_time:
                if stable_label != last_added:
                    sentence += stable_label
                    last_added = stable_label
                    hold_start = time.time()
    
    # =========================
# NLP WORD PREDICTION
# =========================
    current_word = sentence.strip().split(" ")[-1]

    if len(current_word) >= 2:
        suggestions = ngram.suggest_words(current_word, top_k=3)
    else:
        suggestions = []
   

    # =========================
    # DISPLAY
    # =========================
    # =========================
# # =========================
# UI OVERLAY (MODERN STYLE)
# =========================
    overlay = frame.copy()

# Main panel
    cv2.rectangle(overlay, (10, 10), (470, 200), (20, 20, 20), -1)

# Border
    cv2.rectangle(overlay, (10, 10), (470, 200), (0, 255, 255), 2)

# Transparency
    alpha = 0.65
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


# =========================
# ANIMATION: CURSOR BLINK
# =========================
    cursor = "_" if int(time.time() * 2) % 2 == 0 else ""


# =========================
# LETTER DISPLAY
# =========================
    cv2.putText(frame, f"Letter: {stable_label}",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2)

# Confidence %
    cv2.putText(frame, f"{int(conf_val * 100)}%",
                (350, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2)


# =========================
# PROGRESS BAR (CONFIDENCE)
# =========================
    bar_width = int(conf_val * 200)
    cv2.rectangle(frame, (25, 60), (25 + bar_width, 75), (0, 255, 0), -1)
    cv2.rectangle(frame, (25, 60), (225, 75), (255, 255, 255), 1)


# =========================
# TEXT WITH CURSOR
# =========================
    cv2.putText(frame, f"Text: {sentence}{cursor}",
                (25, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 200, 0),
                2)


# =========================
# SUGGESTIONS (HIGHLIGHT TOP)
# =========================
    if suggestions:
    # Top suggestion (highlight)
        cv2.putText(frame, f"{suggestions[0]}",
                    (25, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    3)

    # Other suggestions
        others = ", ".join(suggestions[1:])
        cv2.putText(frame, others,
                    (25, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1)


# =========================
# CONTROLS (CLEAN STYLE)
# =========================
    cv2.putText(frame, "SPACE | D | C | 1=auto | Q",
                (25, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1)
    cv2.imshow("ASL Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    

# Clear full text
    if key == ord('c'):
        sentence = ""
        last_added = ""

# Add SPACE
    if key == ord(' '):
        sentence += " "

# Delete last character
    if key == ord('d'):
        sentence = sentence[:-1]
# AUTO COMPLETE
    if key == ord('1') and suggestions:
        words = sentence.split(" ")
        words[-1] = suggestions[0]
        sentence = " ".join(words)

# Quit app
    if key == ord('q'):
        break
# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()