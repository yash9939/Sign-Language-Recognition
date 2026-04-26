import cv2
import mediapipe as mp
import os
import csv
import time

print("🚀 Script started...")

# =========================
# SETTINGS
# =========================
DATASET_PATH = "landmark_dataset"
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SAMPLES_PER_CLASS = 100

capture_delay = 0.25   # fast capture
countdown_time = 0.1   # almost instant start
pause_time = 2         # time between alphabets

# =========================
# MEDIAPIPE SETUP
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

mp_draw = mp.solutions.drawing_utils

# =========================
# CREATE FOLDER
# =========================
os.makedirs(DATASET_PATH, exist_ok=True)

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not working!")
    exit()

print("✅ Camera opened successfully")

# =========================
# MAIN LOOP
# =========================
for label in CLASSES:

    print(f"\n👉 Collecting data for: {label}")

    file_path = os.path.join(DATASET_PATH, f"{label}.csv")

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)

        count = 0
        last_capture_time = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:

                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    # =========================
                    # NORMALIZATION
                    # =========================
                    x_list = []
                    y_list = []

                    for lm in hand_landmarks.landmark:
                        x_list.append(lm.x)
                        y_list.append(lm.y)

                    base_x = x_list[0]
                    base_y = y_list[0]

                    row = []
                    for x, y in zip(x_list, y_list):
                        row.append(x - base_x)
                        row.append(y - base_y)

                    # =========================
                    # COUNTDOWN + CAPTURE
                    # =========================
                    elapsed = time.time() - start_time

                    if elapsed < countdown_time:
                        cv2.putText(frame, "Get Ready...",
                                    (10,120), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0,0,255), 2)

                    else:
                        current_time = time.time()

                        if current_time - last_capture_time > capture_delay:
                            writer.writerow(row)
                            count += 1
                            last_capture_time = current_time

            # =========================
            # DISPLAY INFO
            # =========================
            cv2.putText(frame, f"Letter: {label}", (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.putText(frame, f"Samples: {count}/{SAMPLES_PER_CLASS}", (10,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            cv2.imshow("Data Collection", frame)

            key = cv2.waitKey(1)

            if key & 0xFF == ord('n'):
                break

            if key & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

            if count >= SAMPLES_PER_CLASS:
                break

    # =========================
    # PAUSE BETWEEN ALPHABETS
    # =========================
    print("⏳ Get ready for next letter...")

    pause_start = time.time()

    while time.time() - pause_start < pause_time:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        cv2.putText(frame, "Next letter coming...",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)

        cv2.imshow("Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()

print("🎉 Data collection completed!")