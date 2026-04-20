from PIL import Image
import time
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from collections import deque

from models.cnn_model import ASL_CNN
from utils.text_builder import update_sentence

# Load model
model = ASL_CNN()
model.load_state_dict(torch.load("saved_models/asl_model.pth", map_location=torch.device('cpu')))
model.eval()

device = torch.device("cpu")
model.to(device)

# Transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# Classes
classes = sorted([
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'del','nothing','space'
])

sentence = ""

# Camera
cap = cv2.VideoCapture(0)

# Control variables
last_time = 0
delay = 1.5

# Prediction smoothing buffer
buffer = deque(maxlen=10)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ROI (hand area)
    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100,100), (400,400), (0,255,0), 2)

    # Preprocess
    img = cv2.resize(roi, (128,128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        output = model(img)
        prob = F.softmax(output, dim=1)
        confidence, pred = torch.max(prob, 1)

    label = classes[pred.item()]

    # Confidence filter
    if confidence.item() < 0.7:
        label = "nothing"

    # Add to buffer
    buffer.append(label)

    # Stabilize prediction
    if len(buffer) == 10:
        label = max(set(buffer), key=buffer.count)

        current_time = time.time()

        # Apply delay
        if current_time - last_time > delay:
            sentence = update_sentence(sentence, label)
            last_time = current_time

    # Display
    cv2.putText(frame, f"{label} ({confidence.item():.2f})", (10,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, sentence, (10,100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("ASL Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()