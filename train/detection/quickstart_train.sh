git clone https://github.com/BertilBraun/Windsurf-Analysis.git
cd Windsurf-Analysis
git checkout production
git pull
pip install -r requirements.txt
python train/detection/train_pose.py --src train/detection/windsurf_dataset --pose train/detection/pose_projects/boom_mast_v1 --base-model yolo11m-pose.pt --device 0 --name pose_seed_500_epochs --epochs 500 --patience 0 --save-period 50



cd train/detection
python train.py --src windsurf_dataset --base-model yolo11m.pt
