import cv2
from ultralytics import YOLO

# =========================
# LOAD MODEL
# =========================
model = YOLO(r".\models\best.pt")

# CLASS MAPPING (YOUR DATASET)
CLASS_NAMES = {
    0: "Helmet",      # With Helmet
    1: "No Helmet"    # Without Helmet
}

# =========================
# INPUT / OUTPUT
# =========================
VIDEO_PATH = r"video1.mp4"
OUTPUT_PATH = "output.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# =========================
# PROCESS
# =========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    alert_active = False

    # YOLO INFERENCE
    results = model(frame, conf=0.4)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # CLASS LOGIC
        if cls == 0:
            label = "Helmet"
            color = (0, 255, 0)   # 🟢 Green
        else:
            label = "No Helmet"
            color = (0, 0, 255)   # 🔴 Red
            alert_active = True

        # DRAW BOUNDING BOX
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # DRAW LABEL
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

    # =========================
    # ALERT BANNER
    # =========================
    if alert_active:
        cv2.rectangle(frame, (0, 0), (520, 60), (0, 0, 255), -1)
        cv2.putText(
            frame,
            "ALERT: Rider Without Helmet!",
            (15, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            3
        )

    out.write(frame)
    cv2.imshow("Helmet Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# =========================
# CLEANUP
# =========================
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Processing complete. Output saved:", OUTPUT_PATH)


