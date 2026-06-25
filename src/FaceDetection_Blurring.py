from ultralytics import YOLO
import cv2 
import numpy as np

def face_detection(model, frame):
    faces_coordinates = []
    faces_masks = []
    faces_idx = []
    results = model.track(frame,persist = True , tracker = 'bytetrack.yaml')
    for result in results:
        if result.boxes is not None and result.boxes.id is not None:
           for box,track_idx in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.id.int().tolist()):
               x1, y1, x2, y2 = map(int, box)
               faces_coordinates.append((x1,y1,x2,y2))
               faces_idx.append(track_idx)
    return faces_coordinates,faces_idx

def blur_faces(frame, faces_coordinates,face_idx):
    if len(face_idx) == 0:
        return frame
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)     
    for track_id,(x1,y1,x2,y2) in zip(face_idx,faces_coordinates):
        child_face = frame[y1:y2,x1:x2]
        h, w = child_face.shape[:2]
        if h == 0 or w == 0:
            continue
        blur = cv2.resize(child_face, (8, 8), interpolation=cv2.INTER_LINEAR)
        frame[y1:y2, x1:x2] = cv2.resize(blur, (w, h), interpolation=cv2.INTER_NEAREST)
    return frame