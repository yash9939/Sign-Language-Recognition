import cv2
import torch
import mediapipe as mp
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import time

# =========================
# CLASSES
# =========================
CLASSES = sorted(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

# =========================
# CONFUSION GROUPS
# =========================
CONFUSION_GROUPS = [
    {'M','N','T'},
    {'A','E','S'},
    {'U','V','W'}
]

def in_same_group(a, b):
    return any(a in g and b in g for g in CONFUSION_GROUPS)

def entropy(probs):
    p = probs + 1e-8
    return float(-(p * p.log()).sum().item())

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
buffer = deque(maxlen=15)
stable_label = ""

sentence = ""
last_added = ""
hold_start = 0
hold_time = 1.2  # seconds

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

            # Extract landmarks
            x_list, y_list = [], []
            for lm in hand_landmarks.landmark:
                x_list.append(lm.x)
                y_list.append(lm.y)

            # Normalize
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

            # =========================
            # PREDICTION (IMPROVED)
            # =========================
            with torch.no_grad():
                logits = model(data)
                probs = F.softmax(logits, dim=1)[0]

            top2_prob, top2_idx = torch.topk(probs, 2)

            p1 = top2_prob[0].item()
            p2 = top2_prob[1].item()

            c1 = CLASSES[top2_idx[0].item()]
            c2 = CLASSES[top2_idx[1].item()]

            gap = p1 - p2
            H = entropy(probs)

            conf_val = p1

            # Base thresholds
            base_thr = 0.40
            gap_thr = 0.10
            ent_thr = 2.2

            # Stricter for confusing letters
            if in_same_group(c1, c2):
                base_thr = 0.55
                gap_thr = 0.18
                ent_thr = 2.0

            if (p1 > base_thr) and (gap > gap_thr) and (H < ent_thr):
                label = c1
            else:
                label = ""

    # =========================
    # SMOOTHING + HOLD LOGIC
    # =========================
    buffer.append(label)

    if len(buffer) == buffer.maxlen:
        most_common = max(set(buffer), key=buffer.count)

        if buffer.count(most_common) >= int(0.7 * buffer.maxlen) and most_common != "":
            
            if stable_label != most_common:
                stable_label = most_common
                hold_start = time.time()

            elif time.time() - hold_start > hold_time:
                if stable_label != last_added:
                    sentence += stable_label
                    last_added = stable_label
                    hold_start = time.time()

    # =========================
    # DISPLAY
    # =========================
    cv2.putText(frame, f"Letter: {stable_label} ({conf_val:.2f})",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.putText(frame, f"Text: {sentence}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2)

    cv2.imshow("ASL Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        sentence = ""
        last_added = ""

    if key == ord('q'):
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()