import os
import warnings

# Suppress technical logs and warnings for a cleaner console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['GLOG_minloglevel'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import time
import math
import numpy as np
from transformers import pipeline
from PIL import Image
import mediapipe as mp

# =====================================================
# Load MobileViTv2
# =====================================================
pipe = pipeline(
    "image-classification",
    model="nguyenkhoa/mobilevitv2_Liveness_detection_v1.0",
    device=-1
)

# =====================================================
# MediaPipe
# =====================================================
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(0,0.6)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# =====================================================
# Helpers
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
    mouth_left=lm[61]
    mouth_right=lm[291]
    upper_lip=lm[13]
    lower_lip=lm[14]
    width=distance(mouth_left,mouth_right)
    height=distance(upper_lip,lower_lip)+1e-6
    return width/height

# -------- TRUE 3D HEAD POSE ----------
def get_head_yaw(lm,w,h):
    face_3d=[]
    face_2d=[]
    idx=[1,33,263,61,291,199]

    for i in idx:
        x,y=int(lm[i].x*w),int(lm[i].y*h)
        face_2d.append([x,y])
        face_3d.append([x,y,lm[i].z*3000])

    face_2d=np.array(face_2d,dtype=np.float64)
    face_3d=np.array(face_3d,dtype=np.float64)

    focal_length=w
    cam_matrix=np.array([[focal_length,0,w/2],
                         [0,focal_length,h/2],
                         [0,0,1]])
    dist_matrix=np.zeros((4,1))

    success,rot_vec,_=cv2.solvePnP(face_3d,face_2d,cam_matrix,dist_matrix)
    rmat,_=cv2.Rodrigues(rot_vec)
    angles,*_=cv2.RQDecomp3x3(rmat)
    return angles[1]

# -------- FACE ALIGNMENT ----------
def align_face(frame,lm,w,h):
    left_eye=lm[33]
    right_eye=lm[263]

    x1,y1=int(left_eye.x*w),int(left_eye.y*h)
    x2,y2=int(right_eye.x*w),int(right_eye.y*h)

    angle=np.degrees(np.arctan2(y2-y1,x2-x1))
    M=cv2.getRotationMatrix2D((w//2,h//2),angle,1)
    return cv2.warpAffine(frame,M,(w,h))

# =====================================================
# STATES
# =====================================================
STATE_SCANNING=0
STATE_BLINK=1
STATE_SMILE=2
STATE_LEFT=3
STATE_RIGHT=4
STATE_DONE=5

state=STATE_SCANNING
last_state_time=time.time()

# =====================================================
# CAMERA
# =====================================================
cap=cv2.VideoCapture(0)
fps=cap.get(cv2.CAP_PROP_FPS)
if fps==0 or fps>60:
    fps=30

frame_interval=max(1,int(fps/2.5))

# =====================================================
# TRACKERS
# =====================================================
live_count=0
spoof_count=0
frame_count=0
start_time=time.time()

score_history=[]
SMOOTH_WINDOW=6
ear_history=[]
nose_history=[]

action_witnessed=False
last_label="Scanning"
last_score=0

TIMEOUT_SEC=30

# =====================================================
# LOOP
# =====================================================
while True:
    ret,frame=cap.read()
    if not ret:
        break

    frame=cv2.flip(frame,1)
    h,w,_=frame.shape
    display_frame=frame.copy()

    if time.time()-start_time>TIMEOUT_SEC:
        break

    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    # ===== FACE MESH =====
    mesh_results=face_mesh.process(rgb)
    if mesh_results.multi_face_landmarks:

        lm=mesh_results.multi_face_landmarks[0].landmark

        # Adaptive EAR
        ear=(eye_aspect_ratio(lm,LEFT_EYE)+
             eye_aspect_ratio(lm,RIGHT_EYE))/2.0

        ear_history.append(ear)
        if len(ear_history)>40:
            ear_history.pop(0)

        baseline=np.mean(ear_history) if len(ear_history)>10 else ear
        dynamic_blink_thresh=baseline*0.75

        sr=smile_ratio(lm)
        yaw=get_head_yaw(lm,w,h)

        # Nose history (micro movement)
        nose=lm[1]
        nose_history.append((nose.x,nose.y))
        if len(nose_history)>100:
            nose_history.pop(0)

        # ===== STATE MACHINE =====
        time_in_state=time.time()-last_state_time
        MIN_ACTION_SEC=5.0

        if state==STATE_SCANNING:
            state=STATE_BLINK
            last_state_time=time.time()
            action_witnessed=False

        elif state==STATE_BLINK:
            if ear<dynamic_blink_thresh:
                action_witnessed=True
            if action_witnessed and time_in_state>MIN_ACTION_SEC:
                state=STATE_SMILE
                last_state_time=time.time()
                action_witnessed=False

        elif state==STATE_SMILE:
            if sr>2.3:
                action_witnessed=True
            if action_witnessed and time_in_state>MIN_ACTION_SEC:
                state=STATE_LEFT
                last_state_time=time.time()
                action_witnessed=False

        elif state==STATE_LEFT:
            if yaw<-10:
                action_witnessed=True
            if action_witnessed and time_in_state>MIN_ACTION_SEC:
                state=STATE_RIGHT
                last_state_time=time.time()
                action_witnessed=False

        elif state==STATE_RIGHT:
            if yaw>10:
                action_witnessed=True
            if action_witnessed and time_in_state>MIN_ACTION_SEC:
                state=STATE_DONE

        # ALIGN FACE BEFORE MODEL
        frame=align_face(frame,lm,w,h)

    # ===== PASSIVE MODEL =====
    face_results=face_detector.process(rgb)
    if face_results.detections:
        detection=face_results.detections[0]
        bbox=detection.location_data.relative_bounding_box

        x1=int(bbox.xmin*w)
        y1=int(bbox.ymin*h)
        bw=int(bbox.width*w)
        bh=int(bbox.height*h)

        expand=1.6
        cx,cy=x1+bw//2,y1+bh//2
        nw,nh=int(bw*expand),int(bh*expand)

        x1=max(0,cx-nw//2)
        y1=max(0,cy-nh//2)
        x2=min(w,cx+nw//2)
        y2=min(h,cy+nh//2)

        # Reject tiny faces
        if (x2-x1)<120:
            continue

        if frame_count%frame_interval==0:
            face_crop=frame[y1:y2,x1:x2]

            if face_crop.size>0:
                pil_img=Image.fromarray(cv2.cvtColor(face_crop,cv2.COLOR_BGR2RGB))
                result=pipe(pil_img)[0]

                raw_score=result["score"]
                lbl_lower=result["label"].lower()
                is_live="live" in lbl_lower or "real" in lbl_lower

                score_history.append(raw_score if is_live else -raw_score)
                if len(score_history)>SMOOTH_WINDOW:
                    score_history.pop(0)

                smooth_score=sum(score_history)/len(score_history)

                last_label="Live" if smooth_score>0 else "Spoof"
                last_score=round(abs(smooth_score),4)

                if smooth_score>0.4:
                    live_count+=1
                elif smooth_score<-0.4:
                    spoof_count+=1

        color=(0,255,0) if last_label=="Live" else (0,0,255)
        cv2.rectangle(display_frame,(x1,y1),(x2,y2),color,2)
        cv2.putText(display_frame,f"{last_label}:{last_score}",
                    (x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    # ===== UI =====
    instructions={
        STATE_SCANNING:"Center Face",
        STATE_BLINK:"Blink",
        STATE_SMILE:"Smile",
        STATE_LEFT:"Turn LEFT",
        STATE_RIGHT:"Turn RIGHT",
        STATE_DONE:"Verifying..."
    }

    cv2.putText(display_frame,instructions[state],
                (20,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)

    cv2.imshow("ULTRA ACCURATE LIVENESS",display_frame)

    if cv2.waitKey(1)&0xFF==ord("q") or state==STATE_DONE:
        break

    frame_count+=1

cap.release()
cv2.destroyAllWindows()

# =====================================================
# FINAL DECISION ENGINE
# =====================================================
print("\n===== SECURITY REPORT =====")

total_ai=live_count+spoof_count
live_ratio=live_count/total_ai if total_ai>0 else 0

movement_valid=False
if len(nose_history)>10:
    nose_arr=np.array(nose_history)
    std_dev=np.std(nose_arr,axis=0)
    if np.sum(std_dev)>0.002:
        movement_valid=True

print("AI Live Ratio:",live_ratio)
print("State:",state)
print("Movement:",movement_valid)

if state==STATE_DONE and live_ratio>0.8 and movement_valid:
    print("FINAL STATUS: VERIFIED HUMAN ✅")
else:
    print("FINAL STATUS: SPOOF ❌")