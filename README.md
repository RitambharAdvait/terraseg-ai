# Startathon-Segmentation-ChallengeEach image has a corresponding segmentation mask file with the same name.

🏷️ Class Labels

The mask images contain pixel values like 0, 100, 200, 300... etc.
We mapped them into class IDs (0-9).

Raw Pixel Value	Class ID	Class Name
0	0	Background
100	1	Trees
200	2	Lush Bushes
300	3	Dry Grass
500	4	Dry Bushes
550	5	Ground Clutter
700	6	Logs
800	7	Rocks
7100	8	Landscape
10000	9	Sky
⚙️ Environment Setup (Anaconda Recommended)
✅ Step 1: Create Conda Environment
conda create -n EDU python=3.10 -y
conda activate EDU

✅ Step 2: Install PyTorch + CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

✅ Step 3: Install Required Packages
pip install numpy tqdm pillow matplotlib
pip install opencv-python==4.9.0.80

🚀 Training the Model
✅ Run Training Script
python train_segmentation.py


During training:

DINOv2 backbone stays frozen (only segmentation head is trained).

Training uses augmentation + normalization.

Best model is saved automatically.

🔥 Output Files Generated

After training:

segmentation_head.pth (final model)

best_segmentation_head.pth (best IoU model)

🧪 Testing / Inference on Test Images
✅ Run Test Script
python test_segmentation.py --model_path segmentation_head.pth


This generates output folder:

predictions/
│
├── masks/           (raw predicted masks 0-9)
├── masks_color/     (colored prediction masks)
├── comparisons/     (input vs gt vs prediction samples)
├── evaluation_metrics.txt
├── per_class_metrics.png

📊 Latest Final Results (Hackathon Submission)
✅ Test Dataset Results (1002 images)

Mean IoU: 0.2174

Mean Accuracy: 0.5691

📌 Output from test script:

Mean IoU:          0.2174
Mean Accuracy:     0.5691
Processed 1002 images

🏆 Best Validation Result Achieved (Training Phase)

From training log:

Best Validation IoU: 0.4209

Best Validation Accuracy: 0.7973

Best Model Saved at:

Epoch 76

File: best_segmentation_head.pth

Training output:

🔥 BEST MODEL SAVED | Epoch 76 | Val IoU = 0.4209
Best Val IoU achieved: 0.4209

📌 Why Test IoU is Lower Than Validation IoU?

Even though our validation IoU reached ~0.42, test IoU dropped (~0.217).
Possible reasons:

✅ Test dataset distribution is different from training/validation
✅ Overfitting to training dataset
✅ Class imbalance (some classes appear less frequently)
✅ Image quality / lighting differences
✅ Mask conversion mismatch risk
✅ Resolution differences between train & test

🛠️ Major Issues Faced During Hackathon (Real Debugging Log)
❌ Issue 1: xFormers Installation Broke Torch

We tried installing xformers:

pip install xformers


It downgraded torch and created version mismatch:

Torch got replaced with incompatible version

Torchvision broke

Training script failed

Error:

RuntimeError: operator torchvision::nms does not exist


✅ Fix:
We reinstalled correct torch/torchvision versions:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

❌ Issue 2: PIL ImportError (DLL load failed)

Error:

ImportError: DLL load failed while importing _imaging: The operating system cannot run %1.


✅ Fix:

pip install pillow==10.4.0

❌ Issue 3: OpenCV cv2 missing / imwrite missing

Errors faced:

ModuleNotFoundError: No module named 'cv2'
AttributeError: module 'cv2' has no attribute 'imwrite'
cv2.__file__ returned None


Cause:

OpenCV package got corrupted or partially installed

✅ Fix:
Reinstalled OpenCV properly:

pip install opencv-python==4.9.0.80

❌ Issue 4: NumPy 2.x compatibility issue with OpenCV

Error:

A module compiled using NumPy 1.x cannot be run in NumPy 2.2.6
ImportError: numpy.core.multiarray failed to import
AttributeError: _ARRAY_API not found


Cause:

OpenCV compiled with NumPy 1.x but environment had NumPy 2.x

✅ Fix:
Downgrade numpy:

pip install "numpy<2"

❌ Issue 5: State_dict Loading Error (Model mismatch)

Error:

Unexpected key(s) in state_dict
size mismatch in stem/block/classifier layers


Cause:

Training head architecture changed (128 vs 256 channels)

Test script model definition didn’t match training model

✅ Fix:
Ensured the same model architecture in both:

train_segmentation.py

test_segmentation.py

🧾 Important Notes

Training IoU and test IoU may differ due to dataset distribution shift.

DINOv2 is a strong feature extractor but segmentation head requires fine-tuning.

Model improvement is possible by:

Training longer

Better augmentations

Using weighted loss

Using larger backbone (vitb14_reg)

📌 Future Improvements

🔹 Train using DINOv2 Base/Large backbone
🔹 Add Class Weighted Loss
🔹 Add Dice Loss + CrossEntropy Combined
🔹 Increase epochs with better learning rate schedule
🔹 Fine-tune backbone instead of freezing
🔹 Use advanced augmentation (MixUp/CutMix for segmentation)

📂 Repository Files
File Name	Purpose
train_segmentation.py	Training script
test_segmentation.py	Evaluation + prediction generation
segmentation_head.pth	Final trained model weights
best_segmentation_head.pth	Best validation IoU model
predictions/	Output results
🧑‍💻 Team Contribution Plan

Hackathon team strategy:

One member handled environment + setup

One member handled training + augmentation

One member handled evaluation + debugging

📌 How to Run Everything Quickly
Train:
python train_segmentation.py

Test:
python test_segmentation.py --model_path segmentation_head.pth

⭐ Final Hackathon Output

✔ Trained DINOv2-based segmentation model
✔ Achieved 0.4209 Val IoU
✔ Generated predictions for 1002 test images
✔ Saved colored segmentation outputs + metrics report

📜 License

This repository is made for hackathon submission and educational use.

🙌 Acknowledgements

Meta AI (DINOv2)

Startathon Hackathon organizers

Torch, Torchvision, OpenCV, NumPy community

🔥 Submission Summary

We successfully built a working semantic segmentation pipeline with:

pretrained transformer backbone

custom segmentation head

full training and testing scripts

predictions saved for all test images

metrics visualization output<img width="1483" height="884" alt="per_class_metrics" src="https://github.com/user-attachments/assets/5788ba06-17c4-4741-9f06-9eb570cb17d9" />
<img width="1200" height="500" alt="training_curves" src="https://github.com/user-attachments/assets/0de33a8c-4304-4596-be6c-30b9be6c6fff" />
<img width="1200" height="500" alt="iou_curves" src="https://github.com/user-attachments/assets/7c672b9f-b031-4c95-93c3-ced15e5def74" />
<img width="1200" height="500" alt="dice_curves" src="https://github.com/user-attachments/assets/a8439572-b93d-41d7-a838-36518afcc1d2" />
<img width="1200" height="1000" alt="all_metrics_curves - Copy" src="https://github.com/user-attachments/assets/be4bdd15-0e8d-41a1-a3d9-2a1b5537ba91" />

