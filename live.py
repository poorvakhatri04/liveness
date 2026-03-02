import cv2
import mediapipe as mp
import numpy as np
import time
import random
import os

class LivenessService:
    def __init__(self):
        self._landmarker = None
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]

        self.MIN_FACE_RATIO = 0.15
        self.MAX_FACE_RATIO = 0.25

    def _get_landmarker(self):
        if self._landmarker is None:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            model_path = os.path.join(os.getcwd(), "face_landmarker.task")

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                num_faces=1
            )

            self._landmarker = vision.FaceLandmarker.create_from_options(options)

        return self._landmarker

    def analyze(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._get_landmarker().detect(mp_image)

        data = {
            "face_detected": False,
            "landmarks": None,
            "distance_status": "unknown",
            "blink": False,
            "smile": False,
            "pose": "center",
            "ear": 0
        }

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            data["face_detected"] = True
            data["landmarks"] = landmarks

            # Distance check
            xs = [lm.x for lm in landmarks]
            face_width = max(xs) - min(xs)

            if face_width < self.MIN_FACE_RATIO:
                data["distance_status"] = "too_far"
            elif face_width > self.MAX_FACE_RATIO:
                data["distance_status"] = "too_close"
            else:
                data["distance_status"] = "good"

            # Blink (EAR)
            def ear(eye):
                p2, p6 = landmarks[eye[1]], landmarks[eye[5]]
                p3, p5 = landmarks[eye[2]], landmarks[eye[4]]
                p1, p4 = landmarks[eye[0]], landmarks[eye[3]]

                v1 = np.linalg.norm([p2.x - p6.x, p2.y - p6.y])
                v2 = np.linalg.norm([p3.x - p5.x, p3.y - p5.y])
                h = np.linalg.norm([p1.x - p4.x, p1.y - p4.y])
                return (v1 + v2) / (2 * h)

            data["ear"] = ear(self.LEFT_EYE)
            if data["ear"] < 0.23:
                data["blink"] = True

            # Head pose
            nose = landmarks[1]
            if nose.x < 0.40:
                data["pose"] = "left"
            elif nose.x > 0.60:
                data["pose"] = "right"
            else:
                data["pose"] = "center"

            # Smile
            if result.face_blendshapes:
                blend = result.face_blendshapes[0]
                sl = sr = 0
                for b in blend:
                    if b.category_name == "mouthSmileLeft":
                        sl = b.score
                    if b.category_name == "mouthSmileRight":
                        sr = b.score
                if (sl + sr) / 2 > 0.4:
                    data["smile"] = True

        return data


def draw_face_guide(frame, min_ratio, max_ratio):
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)

    min_width = int(w * min_ratio)
    max_width = int(w * max_ratio)

    ideal_width = (min_width + max_width) // 2
    radius = ideal_width

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.8, 0)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    frame[mask == 255] = cv2.addWeighted(frame[mask == 255], 1.5, frame[mask == 255], 0, 0)
    cv2.circle(frame, center, radius, (255, 255, 255), 2)

    return frame


def is_aligned(landmarks):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    cx = np.mean(xs)
    cy = np.mean(ys)
    return abs(cx - 0.5) < 0.15 and abs(cy - 0.5) < 0.20


def run_secure_liveness():
    service = LivenessService()
    challenges = ["blink", "smile", "turn_left", "turn_right"]
    random.shuffle(challenges)

    current_step = 0
    waiting = False
    transition_until = 0
    max_time = 30
    start_time = time.time()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not accessible")
        return

    while True:
        ret, original_frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(original_frame, 1)

        small = cv2.resize(frame, (480, 360))
        data = service.analyze(small)

        frame = draw_face_guide(
            frame,
            service.MIN_FACE_RATIO,
            service.MAX_FACE_RATIO
        )

        instruction = "Align your face"

        if data["face_detected"]:

            if data["distance_status"] != "good":
                instruction = f"Move: {data['distance_status']}"
            elif not is_aligned(data["landmarks"]):
                instruction = "Center your face"
            else:
                if current_step < len(challenges):
                    current = challenges[current_step]
                    instruction = f"Do: {current}"

                    if not waiting:

                        action_completed = False

                        if current == "blink" and data["blink"]:
                            action_completed = True
                        elif current == "smile" and data["smile"]:
                            action_completed = True
                        elif current == "turn_left" and data["pose"] == "left":
                            action_completed = True
                        elif current == "turn_right" and data["pose"] == "right":
                            action_completed = True

                        if action_completed:
                            waiting = True
                            transition_until = time.time() + 2
                            instruction = "Good! ✔ Hold still..."
                    if waiting:
                        if time.time() >= transition_until:
                            waiting = False
                            current_step += 1
        
        cv2.putText(frame,"Align your face according to the circle",(20, 30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)

        cv2.putText(frame, instruction, (20, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Secure Liveness Verification", frame)

        if current_step >= len(challenges):
            print("You have done all the challenges!")
            print("Status: Accepted")
            break

        if time.time() - start_time > max_time:
            print("Try again!")
            print("Status: Rejected")
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_secure_liveness()