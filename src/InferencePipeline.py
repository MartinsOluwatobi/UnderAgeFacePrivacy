from ultralytics import YOLO
from src.FaceDetection_Blurring import face_detection, blur_faces
from src.Registry import yaml_loader, model_registry, load_checkpoint
import cv2 as cv
from torchvision import transforms
import torch
from PIL import Image
import os
import urllib.request
import numpy as np

class Inference:
    def __init__(self,config_path):
        self.config = yaml_loader(config_path)
        face_classifier, self.input_size = model_registry(self.config)
        checkpoint = load_checkpoint(self.config)  #torch.load(self.config['model_weight']['model_path'],map_location='cpu')
        face_classifier.load_state_dict(checkpoint['model_state_dict'])
        self.face_classifier= face_classifier
        self.transform = transforms.Compose([transforms.Resize((224,224)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])])
        self.face_classifier.eval()
    def feed(self,vidpath = 0):
        cap = cv.VideoCapture(vidpath)
        model = YOLO("weights/yolov8n-face.pt")
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            faces_coordinates,faces_idx = face_detection(model, frame)
            if count%3 == 0:
               face_detail = {}
            child_faces_coordinates = []
            child_idx = []
            for track_id,(x1, y1, x2, y2) in zip(faces_idx,faces_coordinates):
                if y2 <= y1 or x2 <= x1:
                   continue
                face = frame[y1:y2, x1:x2]
                face = Image.fromarray(cv.cvtColor(face, cv.COLOR_BGR2RGB))
                face =self.transform(face).unsqueeze(0) 
                if track_id not in face_detail.keys():
                   with torch.no_grad():
                      face_group = self.face_classifier(face)
                      face_group = torch.argmax(face_group, dim=1).item()
                      if face_group == 1:
                         child_faces_coordinates.append((x1, y1, x2, y2))
                         child_idx.append(track_id)
                         face_detail[track_id] = face_group
                      else:
                        face_detail[track_id] = 0 
                else:
                     if face_detail[track_id] == 1:
                        child_faces_coordinates.append((x1,y1,x2,y2))
                        child_idx.append(track_id)

            frame = blur_faces(frame, child_faces_coordinates,child_idx)
            count+=1
            cv.imshow("Face blur", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
               break

        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    root = os.getcwd()
    config_file_name = '/config/config.yaml'
    config_path = root+ config_file_name
    infer = Inference(config_path)
    file_name = '/data/Precious Newborn Smiles and Funny Faces! - Cute Baby Videos Compilation.mp4'
    vid_path = root + file_name
    infer.feed(vid_path)