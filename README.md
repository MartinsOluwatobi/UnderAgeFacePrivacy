# UnderAge Face Privacy

Automatically detects and anonymises child faces in video using YOLOv8 face detection, 
ByteTrack tracking, and a ResNet50 age-group classifier.

---

## How it works
Brief explanation of the pipeline:
1. YOLO detects all faces in each frame
2. ByteTrack assigns a persistent tracking ID to each face
3. ResNet50 classifies each new face as child or adult
4. Child faces are pixelated; classification is cached by track ID to avoid 
   re-running the classifier every frame


## Setup

### Requirements
- Python 3.11
- PyTorch
- Ultralytics (YOLOv8)
- OpenCV
- torchvision

Install dependencies:
pip install torch torchvision ultralytics opencv-python pillow


---

## Configuration
All settings are controlled via `config.yaml`:

| Key | Description |
|-----|-------------|
| `model_selection` | which classifier to use (`server_resnet50` or `custom_cnn`) |
| `model_weight.model_path` | path to trained classifier checkpoint |
| `training.batch_size` | batch size used during training |
| `training.learning_rate` | Adam learning rate |
| `training.num_epochs` | number of training epochs |

---

## Training
Dataset: UTKFace — faces labelled with age, gender, ethnicity.
Labels are binarised: age ≤ 13 → child (1), age > 13 → adult (0).

The training was done on Google Collab
1. All script file were exported to google drive
2. Mount Google Drive
3. Run `Training.ipynb`

Trained checkpoint is saved to model.pt.

---

## Running inference
Update `vid_path` in `InferencePipeline.py` to point at your video, then:

python InferencePipeline.py

Press `q` to quit.

---
