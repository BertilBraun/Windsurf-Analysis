# Orientation Classifier Training

This script trains a **YOLOv8 classification model** to recognize the orientation of outdoor video frames in **90° increments** (0°, 90°, 180°, 270°).

## How it works

1. **Dataset build (temporary)**  
   - Reads upright (0°) videos from `--videos` directory.  
   - Randomly samples ~p% of frames (`--sample-prob`).  
   - Assigns each sampled frame to **train/val/test** splits.  
   - Rotates each selected frame into exactly **one orientation class** (0/90/180/270).  
   - Deletes the dataset automatically after training (unless `--keep-dataset` is set).

2. **Model training**  
   - Trains a YOLOv8 classification model (`yolov8n-cls.pt` by default).  
   - Only overrides augmentation options that would corrupt orientation labels:  
     - `degrees=0.0`  
     - `fliplr=0.0`  
     - `flipud=0.0`  
     - `auto_augment="none"`  
   - All other augmentations (color jitter, random erasing, etc.) remain at Ultralytics defaults.

3. **Outputs**  
   - Trained weights:  

     ```
     <outdir>/<run-name>/weights/best.pt
     ```

   - Run logs & metrics under `<outdir>/<run-name>/`.

## Usage

```bash
pip install ultralytics opencv-python tqdm

python train.py \
  --videos ./videos_upright \
  --outdir ./orientation_runs \
  --sample-prob 0.10 \
  --resize-shorter 384 \
  --balance \
  --epochs 30 --batch 64 --imgsz 320
```

## Using the trained model in this repo

The production/local pipelines expect an orientation classifier weight file at:

- `video_processing/inference/weights/orientation_fixer/best.pt`

So after training, copy:

- `<outdir>/<run-name>/weights/best.pt` → `video_processing/inference/weights/orientation_fixer/best.pt`

Then redeploy Modal (`python video_processing/deploy.py`) if you want the change in production.
