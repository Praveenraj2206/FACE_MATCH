import cv2
import numpy as np
from deepface import DeepFace
from collections import deque

# -------------------- Helper --------------------
def l2_normalize(x):
    return x / np.linalg.norm(x)

# -------------------- Load reference images --------------------
print("[INFO] Loading reference images...")
reference_paths = ["me1.jpg", "me2.jpg", "me3.jpg"]  # add more if you want
ref_embeddings = []

for path in reference_paths:
    ref_img = cv2.imread(path)
    if ref_img is None:
        raise FileNotFoundError(f"Reference image '{path}' not found!")
    ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

    emb = DeepFace.represent(
        ref_rgb,
        model_name="VGG-Face",          # you can also try "ArcFace" for better stability
        detector_backend="retinaface",  # more reliable
        enforce_detection=True
    )[0]["embedding"]

    ref_embeddings.append(l2_normalize(np.array(emb)))

# Average embedding
ref_embedding = np.mean(ref_embeddings, axis=0)
ref_embedding = l2_normalize(ref_embedding)
print("[INFO] Reference embedding computed.")

# -------------------- Initialize webcam --------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("[INFO] Starting camera... Press 'q' to quit.")

# -------------------- Stability logic --------------------
MATCH_THRESHOLD = 0.95   # start high, adjust based on [DEBUG] distances
window_size = 5          # smooth over last N frames
match_history = deque(maxlen=window_size)

# -------------------- Main loop --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        embedding = DeepFace.represent(
            rgb_frame,
            model_name="VGG-Face",
            detector_backend="retinaface",
            enforce_detection=True
        )[0]["embedding"]

        embedding = l2_normalize(np.array(embedding))

        # Euclidean distance
        dist = np.linalg.norm(ref_embedding - embedding)
        print(f"[DEBUG] Distance: {dist:.4f}")

        # Decide match for this frame
        is_match = dist < MATCH_THRESHOLD
        match_history.append(is_match)

        # Smooth decision over last N frames
        if sum(match_history) > len(match_history) // 2:
            text = "MATCH FOUND"
            color = (0, 255, 0)
        else:
            text = "NO MATCH"
            color = (0, 0, 255)

        cv2.putText(frame, text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    except Exception as e:
        print(f"[WARN] Face not detected: {e}")
        cv2.putText(frame, "NO FACE DETECTED", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Face Verification - VGG-Face", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -------------------- Cleanup --------------------
cap.release()
cv2.destroyAllWindows()
