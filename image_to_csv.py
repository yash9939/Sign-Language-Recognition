import cv2
import mediapipe as mp
import os
import csv

# =========================
# PATHS
# =========================
INPUT_DATASET = os.path.join(os.getcwd(), "Dataset", "train")
print("Dataset path:", INPUT_DATASET)
print("Folders found:", os.listdir(INPUT_DATASET))       # your image dataset
OUTPUT_DATASET = "landmark_dataset"  # output CSV folder

# =========================
# MEDIAPIPE SETUP
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# =========================
# CREATE OUTPUT FOLDER
# =========================
os.makedirs(OUTPUT_DATASET, exist_ok=True)

# =========================
# CONVERT IMAGES → CSV
# =========================
for label in os.listdir(INPUT_DATASET):
    label_path = os.path.join(INPUT_DATASET, label)

    if not os.path.isdir(label_path):
        continue

    print(f"\n👉 Processing: {label}")

    csv_path = os.path.join(OUTPUT_DATASET, f"{label}.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        count = 0

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)

            image = cv2.imread(img_path)
            if image is None:
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            # =========================
            # IF HAND DETECTED
            # =========================
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:

                    # NORMALIZATION (IMPORTANT)
                    x_list = []
                    y_list = []

                    for lm in hand_landmarks.landmark:
                        x_list.append(lm.x)
                        y_list.append(lm.y)

                    base_x = x_list[0]  # wrist
                    base_y = y_list[0]

                    row = []
                    for x, y in zip(x_list, y_list):
                        row.append(x - base_x)
                        row.append(y - base_y)

                    writer.writerow(row)
                    count += 1

        print(f"✅ Saved {count} samples for {label}")

print("\n🎉 ALL IMAGES CONVERTED TO CSV!")