# Detection Weights

This folder expects two Ultralytics YOLO checkpoints:

- `ppe_model.pt`
- `crack_model.pt`

## Train both models

From the project root:

```bash
python detection/ai_models/train_models.py --base-model yolov8n.pt --epochs 20 --imgsz 512 --batch 4 --device auto
```

If you want an even lighter test run first, try:

```bash
python detection/ai_models/train_models.py --epochs 5 --imgsz 416 --batch 2
```

The script trains on:

- `dataset/ppe_dataset/data.yaml`
- `dataset/crack_dataset/data.yaml`

After training, it copies the best checkpoint from each run into this folder with the exact filenames the app loads.
