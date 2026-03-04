git clone https://github.com/BertilBraun/Windsurf-Analysis.git
cd Windsurf-Analysis
pip install -r requirements.txt
python train/detection/train_pose.py --src train/detection/windsurf_dataset --pose train/detection/pose_projects/boom_mast_v1 --base-model yolo11m-pose.pt --device 0 --name pose_seed_700_epochs --epochs 700 --patience 0 --save-period 200



cd train/detection
python train.py --src windsurf_dataset --base-model yolo11m.pt


python train/detection/mine_hard_windows.py --videos "new_training_vids" --out "hard_windows.txt"
