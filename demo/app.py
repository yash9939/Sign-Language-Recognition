from PIL import Image
import time
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from collections import deque
import mediapipe as mp

from models.cnn_model import ASL_CNN
from utils.text_builder import update_sentence

# =========================
# LOAD MODEL mmm
# =========================
model = ASL_CNN()
model.load_state_dict(torch.load("saved_models/asl_model.pth", map_location=torch.device('cpu')))
model.eval()

device = torch.device("cpu")
model.to(device)

# =========================
# TRANSFORM (match training)
# =========================
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# =========================
# IMPORTANT: CORRECT CLASS ORDER
# =========================
classes = sorted([
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z'
])

sentence = ""

# =========================
# MEDIAPIPE SETUP
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

# =========================
# CONTROL VARIABLES
# =========================
buffer = deque(maxlen=10)
stable_label = ""
hold_start_time = 0
hold_time_required = 2.0  # seconds

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for natural interaction
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    label = "nothing"
    conf_val = 0.0

    # =========================
    # HAND DETECTION
    # =========================
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_list, y_list = [], []

            for lm in hand_landmarks.landmark:
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)

            # Bigger padding for full hand
            padding = 80
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)

            roi = frame[y_min:y_max, x_min:x_max]

            if roi.size != 0:
                img = cv2.resize(roi, (128,128))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(img)
                    prob = F.softmax(output, dim=1)
                    confidence, pred = torch.max(prob, 1)

                conf_val = confidence.item()

                # Strong confidence filtering
                if conf_val > 0.95:
                    label = classes[pred.item()]
                else:
                    label = ""

    # =========================
    # TEMPORAL SMOOTHING + HOLD LOGIC
    # =========================
    buffer.append(label)

    if len(buffer) == buffer.maxlen:
        most_common = max(set(buffer), key=buffer.count)

        # Only accept very stable predictions
        if buffer.count(most_common) >= 8 and most_common != "":
            
            # New gesture detected
            if stable_label != most_common:
                stable_label = most_common
                hold_start_time = time.time()

            # Gesture held long enough
            elif time.time() - hold_start_time > hold_time_required:
                sentence = update_sentence(sentence, stable_label)
                hold_start_time = time.time()

        else:
            stable_label = ""

    # =========================
    # DISPLAY
    # =========================
    cv2.putText(frame, f"{label} ({conf_val:.2f})", (10,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, sentence, (10,100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("ASL Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()