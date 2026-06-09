"""Train the PPE and crack YOLO models used by the mining safety app.

This script trains two separate detectors from the bundled Roboflow exports:
- dataset/ppe_dataset/data.yaml
- dataset/crack_dataset/data.yaml

After training, the best checkpoint from each run is copied to:
- detection/ai_models/ppe_model.pt
- detection/ai_models/crack_model.pt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_MODEL = "yolov8n.pt"
DEFAULT_EPOCHS = 20
DEFAULT_IMGSZ = 512
DEFAULT_BATCH = 4
DEFAULT_WORKERS = 2
DEFAULT_PATIENCE = 10

PPE_DATA_YAML = PROJECT_ROOT / "dataset" / "ppe_dataset" / "data.yaml"
CRACK_DATA_YAML = PROJECT_ROOT / "dataset" / "crack_dataset" / "data.yaml"


def resolve_device(requested_device: str | None) -> str:
    if requested_device and requested_device.lower() != "auto":
        return requested_device

    if torch.cuda.is_available():
        return "0"

    return "cpu"


def train_and_export(*, data_yaml: Path, output_filename: str, run_name: str, base_model: str,
                     epochs: int, imgsz: int, batch: int, workers: int, patience: int,
                     device: str | None) -> Path:
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    resolved_device = resolve_device(device)
    model = YOLO(base_model)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        patience=patience,
        device=resolved_device,
        amp=resolved_device != "cpu",
        cache=False,
        project=str(PROJECT_ROOT / "runs"),
        name=run_name,
        exist_ok=True,
    )

    trainer = getattr(model, "trainer", None)
    save_dir = Path(trainer.save_dir) if trainer and getattr(trainer, "save_dir", None) else None

    if save_dir is None:
        raise RuntimeError("Ultralytics did not expose a save directory for the completed run.")

    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Training completed, but best weights were not found: {best_weights}")

    target_path = MODEL_DIR / output_filename
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, target_path)
    return target_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and export mining safety YOLO models.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                        help="Ultralytics base model to fine-tune, for example yolov8n.pt.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Training epochs for each model. Default: {DEFAULT_EPOCHS}.")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                        help=f"Training image size. Smaller is faster/cooler. Default: {DEFAULT_IMGSZ}.")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                        help=f"Batch size. Smaller uses less VRAM and CPU. Default: {DEFAULT_BATCH}.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Dataloader workers. Default: {DEFAULT_WORKERS}.")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE,
                        help=f"Early stopping patience. Default: {DEFAULT_PATIENCE}.")
    parser.add_argument("--device", default="auto",
                        help='Torch device, for example cpu, 0, 0,1, mps, or auto. Default: auto.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ppe_target = train_and_export(
        data_yaml=PPE_DATA_YAML,
        output_filename="ppe_model.pt",
        run_name="ppe",
        base_model=args.base_model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        device=args.device,
    )

    crack_target = train_and_export(
        data_yaml=CRACK_DATA_YAML,
        output_filename="crack_model.pt",
        run_name="crack",
        base_model=args.base_model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        device=args.device,
    )

    print(f"Saved PPE model to: {ppe_target}")
    print(f"Saved crack model to: {crack_target}")
    print(f"Training device: {resolve_device(args.device)}")


if __name__ == "__main__":
    main()
