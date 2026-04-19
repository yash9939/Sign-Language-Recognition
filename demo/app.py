import cv2
import torch
from torchvision import transforms

from models.cnn_model import ASL_CNN
from utils.text_builder import update_sentence

# Load model
model = ASL_CNN()
model.load_state_dict(torch.load("models/asl_model.pth"))
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# Class labels (VERY IMPORTANT ORDER)
classes = sorted([
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'del','nothing','space'
])

sentence = ""

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # ROI (you can improve this later)
    img = cv2.resize(frame, (128,128))
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)
        label = classes[pred.item()]

    # Update sentence
    sentence = update_sentence(sentence, label)

    # Display
    cv2.putText(frame, f"{label}", (10,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, sentence, (10,100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("ASL Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()