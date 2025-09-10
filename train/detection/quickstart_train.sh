git clone https://github.com/BertilBraun/Windsurf-Analysis.git
cd Windsurf-Analysis
git checkout production
git pull
pip install -r requirements.txt
cd train/detection
python train.py --src windsurf_dataset --base-model yolo11m.pt