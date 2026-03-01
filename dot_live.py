import cv2
import mediapipe as mp
import numpy as np
import time
import os
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.getcwd(), "face_landmarker.task")
DOT_SPEED = 200
CHALLENGE_DURATION = 20  
REQUIRED_SIMILARITY = 0.65
FRAME_WIDTH = 640
FRAME_HEIGHT = 640

CLR_BG = (15, 15, 15)
CLR_ACCENT = (0, 255, 127)
CLR_DOT = (0, 71, 255) 
CLR_WHITE = (245, 245, 245)
CLR_RED = (0, 0, 255)

def get_landmarker():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)

def draw_styled_text(img, text, pos, font_scale=0.8, thickness=1, color=CLR_WHITE):
    cv2.putText(img, text, (pos[0]+2, pos[1]+2), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0,0,0), thickness)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness)

def draw_progress_bar(img, progress, w):
    bar_w = int(w * 0.8)
    bar_h = 10
    x = int(w * 0.1)
    y = 60
    
    # Background
    cv2.rectangle(img, (x, y), (x + bar_w, y + bar_h), (50, 50, 50), -1, cv2.LINE_AA)
    # Fill
    fill_w = int(bar_w * progress)
    cv2.rectangle(img, (x, y), (x + fill_w, y + bar_h), CLR_ACCENT, -1, cv2.LINE_AA)

def run_dot_liveness():
    landmarker = get_landmarker()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    t_start = time.time()
    A, B = w // 3, h // 3
    a, b = 1.1, 1.3
    delta = math.pi / 2
    prev_iris = None
    prev_dot = None
    similarity_scores = []
    
    print("Starting Dot Tracking Liveness...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        elapsed = time.time() - t_start
        
        if elapsed > CHALLENGE_DURATION:
            break

        scale = DOT_SPEED / (A * a)
        t_scaled = elapsed * scale * 2.0 
        
        dot_x = int(w // 2 + A * math.sin(a * t_scaled))
        dot_y = int(h // 2 + B * math.sin(b * t_scaled + delta))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        # Draw the target dot (normal red circle)
        cv2.circle(frame, (dot_x, dot_y), 15, CLR_DOT, -1, cv2.LINE_AA)

        # Process Landmarks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Performance optimization: we could resize RGB frame here for faster detection
        # but modern CPUs handle 720p landmarks reasonably well.
        result = landmarker.detect(mp_image)

        if result.face_landmarks:
            marks = result.face_landmarks[0]
            
            # Use both irises for robustness
            # Left Iris: 468, Right Iris: 473
            l_iris = marks[468]
            r_iris = marks[473]
            
            # Average iris center in pixel coordinates
            avg_iris_x = (l_iris.x + r_iris.x) / 2 * w
            avg_iris_y = (l_iris.y + r_iris.y) / 2 * h
            
            # Visualize detected irises (small dots)
            cv2.circle(frame, (int(l_iris.x * w), int(l_iris.y * h)), 2, -1)
            cv2.circle(frame, (int(r_iris.x * w), int(r_iris.y * h)), 2, -1)

            if prev_iris is not None and prev_dot is not None:
                # Calculate Movement Vectors
                iris_vec = np.array([avg_iris_x - prev_iris[0], avg_iris_y - prev_iris[1]])
                dot_vec = np.array([dot_x - prev_dot[0], dot_y - prev_dot[1]])
                
                # Check for significant movement to avoid noise from jitter
                if np.linalg.norm(dot_vec) > 1.0:
                    dot_norm = np.linalg.norm(dot_vec)
                    iris_norm = np.linalg.norm(iris_vec)
                    
                    if iris_norm > 0.5:
                        # Cosine similarity
                        similarity = np.dot(iris_vec, dot_vec) / (iris_norm * dot_norm)
                        # We only care if the movement is in the same direction
                        similarity_scores.append(max(0, similarity))

            prev_iris = (avg_iris_x, avg_iris_y)
            prev_dot = (dot_x, dot_y)
        else:
            draw_styled_text(frame, "Face Not Detected!", (w//2 - 150, h//2), color=CLR_RED)

        # Instruction & Progress
        draw_styled_text(frame, "FOCUS ON THE MOVING DOT", (w//2 - 200, 40), font_scale=0.9, color=CLR_ACCENT)
        draw_progress_bar(frame, elapsed / CHALLENGE_DURATION, w)

        cv2.imshow("Premium Dot Liveness", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if len(similarity_scores) < 30:
        print("\nRESULT: REJECTED")
        print("Reason: Insufficient tracking data.")
        return False

    score = np.mean(similarity_scores)
    robust_score = np.median(similarity_scores)
    
    print(f"\nTracking Consistency Score: {robust_score:.4f}")
    
    if robust_score > REQUIRED_SIMILARITY:
        print("STATUS: VERIFIED (Live User)")
        return True
    else:
        print("STATUS: REJECTED (Suspicious Movement)")
        return False

if __name__ == "__main__":
    run_dot_liveness()