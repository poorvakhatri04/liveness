import os
import cv2
import time
import math
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import random

# =====================================================
# LOAD MINIFASNET
# =====================================================
mini_session = ort.InferenceSession("minifasnet.onnx")

def miniFASNet_predict(face_crop):

    img = cv2.resize(face_crop, (128,128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # IMPORTANT
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img,(2,0,1))
    img = np.expand_dims(img,axis=0)

    input_name = mini_session.get_inputs()[0].name
    output = mini_session.run(None,{input_name:img})[0]

    # FIXED OUTPUT ORDER (common MiniFASNetV2 format)
    live_score  = float(output[0][1])
    spoof_score = float(output[0][0])

    return live_score - spoof_score

# =====================================================
# MEDIAPIPE
# =====================================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# =====================================================
# HELPERS
# =====================================================
def distance(p1,p2):
    return math.hypot(p1.x-p2.x,p1.y-p2.y)

LEFT_EYE=[33,160,158,133,153,144]
RIGHT_EYE=[362,385,387,263,373,380]

def eye_aspect_ratio(lm,eye):
    v1=distance(lm[eye[1]],lm[eye[5]])
    v2=distance(lm[eye[2]],lm[eye[4]])
    h=distance(lm[eye[0]],lm[eye[3]])
    return (v1+v2)/(2.0*h)

def smile_ratio(lm):
    width=distance(lm[61],lm[291])
    height=distance(lm[13],lm[14])+1e-6
    return width/height

def get_head_yaw(lm,w,h):
    idx=[1,33,263,61,291,199]
    face_2d=np.array([[lm[i].x*w,lm[i].y*h] for i in idx],dtype=np.float64)
    face_3d=np.array([[lm[i].x*w,lm[i].y*h,lm[i].z*2000] for i in idx],dtype=np.float64)

    focal=w*0.9
    cam=np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1]],dtype=np.float64)
    dist=np.zeros((4,1))

    success,rot,_=cv2.solvePnP(face_3d,face_2d,cam,dist)
    if not success:
        return 0.0

    rmat,_=cv2.Rodrigues(rot)
    sy=math.sqrt(rmat[0,0]**2+rmat[1,0]**2)
    yaw=math.degrees(math.atan2(-rmat[2,0],sy)) if sy>1e-6 else 0.0
    return yaw

def align_face(frame,lm,w,h):
    # Use eye corners to find rotation angle
    x1,y1=int(lm[33].x*w),int(lm[33].y*h)
    x2,y2=int(lm[263].x*w),int(lm[263].y*h)
    angle=np.degrees(np.arctan2(y2-y1,x2-x1))
    
    # FIX: Use nose tip as rotation center to keep face position stable in the frame
    center=(int(lm[1].x*w), int(lm[1].y*h))
    
    M=cv2.getRotationMatrix2D(center,angle,1)
    return cv2.warpAffine(frame,M,(w,h))

def get_texture_score(face_crop):
    gray=cv2.cvtColor(face_crop,cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray,cv2.CV_64F).var()

# =====================================================
# STATES
# =====================================================
STATE_SCANNING=0
STATE_BLINK=1
STATE_SMILE=2
STATE_LEFT=3
STATE_RIGHT=4
STATE_VERIFYING=5
STATE_DONE=6

INSTRUCTIONS = {
    STATE_SCANNING: "Face the Camera",
    STATE_BLINK: "Blink Your Eyes",
    STATE_SMILE: "Smile Big",
    STATE_LEFT: "Turn Head Left",
    STATE_RIGHT: "Turn Head Right",
    STATE_VERIFYING: "Verifying Liveness...",
    STATE_DONE: "Verification Complete!"
}

# GUIDED ACTION SEQUENCE (Randomized)
ACTION_SEQUENCE = [STATE_BLINK, STATE_SMILE, STATE_LEFT, STATE_RIGHT]
random.shuffle(ACTION_SEQUENCE)
action_idx = 0

state=STATE_SCANNING
last_state_time=time.time()
# Track AI performance per action
action_stats = {s: {"live": 0, "spoof": 0} for s in ACTION_SEQUENCE}

# =====================================================
# CAMERA
# =====================================================
cap=cv2.VideoCapture(0)
fps=cap.get(cv2.CAP_PROP_FPS)
if fps==0 or fps>60: fps=30

# =====================================================
# TRACKERS
# =====================================================
live_count=0
spoof_count=0
frame_count=0
start_time=time.time()

score_history=[]
texture_history=[]
ear_history=[]
nose_history=[]
distance_history=[]
SMOOTH_WINDOW=6

last_label="Scanning"
last_score=0
last_texture=0
dist_m=0

TIMEOUT_SEC=30

# =====================================================
# MAIN LOOP
# =====================================================
while True:
    ret,frame=cap.read()
    if not ret: break

    frame=cv2.flip(frame,1)
    h,w,_=frame.shape
    display_frame=frame.copy()

    if time.time()-start_time>TIMEOUT_SEC:
        break

    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    mesh_results=face_mesh.process(rgb)

    if mesh_results.multi_face_landmarks:
        lm=mesh_results.multi_face_landmarks[0].landmark

        # Distance Estimation (Outer eye corners distance used as baseline ~95mm)
        eye_dist_px = math.hypot((lm[33].x - lm[263].x) * w, (lm[33].y - lm[263].y) * h)
        focal_length = w * 0.9  # Rough estimate for webcams
        current_dist_m = (95 * focal_length) / (eye_dist_px * 1000 + 1e-6)
        distance_history.append(current_dist_m)
        if len(distance_history)>20: distance_history.pop(0)
        dist_m = np.mean(distance_history)

        ear=(eye_aspect_ratio(lm,LEFT_EYE)+eye_aspect_ratio(lm,RIGHT_EYE))/2
        ear_history.append(ear)
        if len(ear_history)>40: ear_history.pop(0)
        baseline=np.mean(ear_history) if len(ear_history)>10 else ear
        dynamic_blink_thresh=baseline*0.75

        sr=smile_ratio(lm)
        yaw=get_head_yaw(lm,w,h)

        nose=lm[1]
        nose_history.append((nose.x,nose.y))
        if len(nose_history)>100: nose_history.pop(0)

        motion_now=np.std(np.array(nose_history[-10:])) if len(nose_history)>10 else 0

        time_in_state=time.time()-last_state_time
        MIN_ACTION_SEC=3.0

        # =====================================================
        # RANDOMIZED STATE TRANSITIONS
        # =====================================================
        if state == STATE_SCANNING and time_in_state > 2.0 and motion_now > 0.0001:
            state = ACTION_SEQUENCE[action_idx]
            last_state_time = time.time()
        elif state in ACTION_SEQUENCE and time_in_state > MIN_ACTION_SEC:
            action_complete = False
            if state == STATE_BLINK and ear < dynamic_blink_thresh: action_complete = True
            elif state == STATE_SMILE and sr > 2.5: action_complete = True
            elif state == STATE_LEFT and yaw < -10: action_complete = True
            elif state == STATE_RIGHT and yaw > 10: action_complete = True
            
            if action_complete:
                action_idx += 1
                if action_idx < len(ACTION_SEQUENCE):
                    state = ACTION_SEQUENCE[action_idx]
                else:
                    state = STATE_VERIFYING
                last_state_time = time.time()
        elif state == STATE_VERIFYING and time_in_state > 5.0:
            state = STATE_DONE
            last_state_time = time.time()

        aligned_frame = align_face(frame,lm,w,h)

        xs=[p.x for p in lm]; ys=[p.y for p in lm]
        x1=int(min(xs)*w); y1=int(min(ys)*h)
        x2=int(max(xs)*w); y2=int(max(ys)*h)

        cx,cy=(x1+x2)//2,(y1+y2)//2
        # FIX: Enforce square crop to prevent stretching/distortion in MiniFASNet
        side = int(max(x2-x1, y2-y1) * 1.4)
        x1=max(0,cx-side//2); y1=max(0,cy-side//2)
        x2=min(w,cx+side//2); y2=min(h,cy+side//2)

        run_ai=(state!=STATE_SCANNING) and (frame_count%2==0)

        if (x2-x1)>60 and run_ai:
            face_crop=aligned_frame[y1:y2,x1:x2]

            if face_crop.size>0:
                t_score=get_texture_score(face_crop)
                texture_history.append(t_score)
                if len(texture_history)>10: texture_history.pop(0)
                last_texture=round(t_score,1)

                mini_score = miniFASNet_predict(face_crop)
                score_history.append(mini_score)

                if len(score_history)>SMOOTH_WINDOW:
                    score_history.pop(0)

                # ===== PRO TEMPORAL SMOOTHING =====
                weights = np.linspace(1,2,len(score_history))
                smooth_score = np.average(score_history,weights=weights)

                # Motion bonus (anti replay)
                smooth_score += motion_now * 20

                # Distance penalty
                if dist_m > 0.8:
                    smooth_score *= 0.8

                # ===== CONFIDENCE BAND =====
                if smooth_score > 0.08:
                    last_label = "Live"
                elif smooth_score < -0.08:
                    last_label = "Spoof"
                else:
                    last_label = "Checking"

                last_score = round(abs(smooth_score),4)

                if smooth_score>0.02:
                    live_count+=1
                    if state in action_stats: action_stats[state]["live"] += 1
                elif smooth_score<-0.05:
                    spoof_count+=1
                    if state in action_stats: action_stats[state]["spoof"] += 1

        color=(0,255,0) if last_label=="Live" else (0,0,255)
        cv2.rectangle(display_frame,(x1,y1),(x2,y2),color,2)
        dist_text = f"Dist: {dist_m:.2f}m"
        cv2.putText(display_frame,
                    f"AI:{last_label}({last_score}) TX:{last_texture} {dist_text}",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

    cv2.putText(display_frame,INSTRUCTIONS[state],
                (20,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)

    cv2.imshow("MULTI-LAYER LIVENESS",display_frame)

    if cv2.waitKey(1)&0xFF==ord("q"): break
    if state==STATE_DONE and time.time()-last_state_time>3.5: break

    frame_count+=1

cap.release()
cv2.destroyAllWindows()

print("\n===== FINAL SECURITY REPORT =====")

total_ai=live_count+spoof_count
live_ratio=live_count/total_ai if total_ai>0 else 0
avg_texture=np.mean(texture_history) if texture_history else 0

movement_valid=False
if len(nose_history)>10:
    std_dev=np.std(np.array(nose_history),axis=0)
    if np.sum(std_dev)>0.005: movement_valid=True

# FINAL EVALUATION (Adjusted for glasses support)
ai_ok=(live_ratio>0.8)        # Lowered from 0.7
texture_ok=(avg_texture>15)   # Lowered from 22 to handle glint/reflections
state_ok=(state in [STATE_DONE,STATE_VERIFYING])
move_ok=movement_valid
distance_ok=(dist_m <= 1.2) # Allow up to 1m with 20% buffer

print(f"AI VERIFICATION: {'PASSED' if ai_ok else 'FAILED'}")
print(f"TEXTURE ANALYSIS: {'PASSED' if texture_ok else 'FAILED'}")

# Calculate AI performance for actions
print("\n--- ACTION AI VERIFICATION ---")
actions_passed_ai = 0
for s in ACTION_SEQUENCE:
    name = INSTRUCTIONS[s]
    stats = action_stats[s]
    total = stats["live"] + stats["spoof"]
    ratio = stats["live"] / total if total > 0 else 0
    passed = ratio > 0.6
    if passed: actions_passed_ai += 1
    print(f"{name:18}: { 'PASSED' if passed else 'FAILED' } (Live Ratio: {ratio:.2f})")

all_actions_ai_ok = (actions_passed_ai == len(ACTION_SEQUENCE))

print(f"\nGUIDED ACTIONS: {'PASSED' if state_ok else 'FAILED'}")
print(f"MICRO-MOVEMENT: {'PASSED' if move_ok else 'FAILED'}")
print(f"DISTANCE CHECK: {'PASSED' if distance_ok else 'FAILED'} ({dist_m:.2f}m)")

if ai_ok and texture_ok and state_ok and move_ok and distance_ok and all_actions_ai_ok:
    print("\n✅ STATUS: VERIFIED HUMAN")
else:
    print("\n❌ STATUS: REJECTED")
