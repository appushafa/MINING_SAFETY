from pathlib import Path

import cv2
from django.conf import settings

MODEL_DIR = Path(__file__).resolve().parent
PPE_MODEL_PATH = MODEL_DIR / 'ppe_model.pt'

_ppe_model = None


def _normalize_label(label):
    return str(label).strip().lower().replace(" ", "_").replace("-", "_")


def _model_label_set(model):
    if model is None:
        return set()

    return {_normalize_label(name) for name in model.names.values()}


def _load_model(model_path):
    if not model_path.exists():
        return None

    from ultralytics import YOLO  # lazy import — avoids loading torch/matplotlib at startup
    return YOLO(str(model_path))


def _get_ppe_model():
    global _ppe_model

    if _ppe_model is None:
        _ppe_model = _load_model(PPE_MODEL_PATH)

    return _ppe_model


def _ppe_support_map(ppe_model):
    labels = _model_label_set(ppe_model)

    return {
        'helmet': 'helmet' in labels,
        'gloves': 'gloves' in labels,
        'jacket': bool(labels.intersection({
            'vest',
            'safety_vest',
            'jacket',
            'reflective_vest',
        })),
        'shoes': bool(labels.intersection({
            'boots',
            'shoes',
            'safety_boots',
        })),
        'no_helmet': 'no_helmet' in labels,
    }


def _labels_from_results(results, model):
    labels = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            labels.append(_normalize_label(model.names[cls]))

    return labels


def _result_path_for(image_path):
    image_path = Path(image_path)
    result_dir = Path(settings.MEDIA_ROOT) / 'results'
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir / image_path.name


def process_image(image_path):

    image_path = Path(image_path)
    image = cv2.imread(str(image_path))

    if image is None:
        return {
            'helmet': False,
            'gloves': False,
            'jacket': False,
            'shoes': False,
            'crack': False,
            'helmet_missing': False,
            'supported_ppe': {
                'helmet': False,
                'gloves': False,
                'jacket': False,
                'shoes': False,
                'no_helmet': False,
            },
            'status': 'IMAGE LOAD FAILED',
        }

    ppe_model = _get_ppe_model()

    if ppe_model is None:
        result_path = _result_path_for(image_path)
        cv2.imwrite(str(result_path), image)
        return {
            'helmet': False,
            'gloves': False,
            'jacket': False,
            'shoes': False,
            'crack': False,
            'helmet_missing': False,
            'supported_ppe': {
                'helmet': False,
                'gloves': False,
                'jacket': False,
                'shoes': False,
                'no_helmet': False,
            },
            'status': 'MODEL FILES MISSING',
            'result_image': str(result_path.relative_to(settings.MEDIA_ROOT)).replace('\\', '/'),
        }

    ppe_results = ppe_model(image)

    ppe_labels = _labels_from_results(ppe_results, ppe_model)

    supported_ppe = _ppe_support_map(ppe_model)

    helmet = 'helmet' in ppe_labels
    gloves = 'gloves' in ppe_labels
    jacket = bool(set(ppe_labels).intersection({
        'vest',
        'safety_vest',
        'jacket',
        'reflective_vest',
    }))
    shoes = bool(set(ppe_labels).intersection({
        'boots',
        'shoes',
        'safety_boots',
    }))

    helmet_missing = 'no_helmet' in ppe_labels
    crack = False

    required_ppe_found = helmet and jacket and not helmet_missing
    status = "ALLOWED FOR MINING" if required_ppe_found else "NOT ALLOWED FOR MINING"

    annotated_frame = ppe_results[0].plot() if ppe_results else image

    result_path = _result_path_for(image_path)

    cv2.imwrite(str(result_path), annotated_frame)

    return {
        "helmet": helmet,
        "gloves": gloves,
        "jacket": jacket,
        "shoes": shoes,
        "crack": crack,
        "helmet_missing": helmet_missing,
        "supported_ppe": supported_ppe,
        "ppe_labels": ppe_labels,
        "status": status,
        "result_image": str(result_path.relative_to(settings.MEDIA_ROOT)).replace('\\', '/')
    }


def process_frame(image):
    """Processes a single BGR OpenCV frame from a video stream and returns the annotated frame and status."""
    if image is None:
        return None, {
            'helmet': False,
            'gloves': False,
            'jacket': False,
            'shoes': False,
            'crack': False,
            'helmet_missing': False,
            'supported_ppe': {
                'helmet': False,
                'gloves': False,
                'jacket': False,
                'shoes': False,
                'no_helmet': False,
            },
            'status': 'IMAGE LOAD FAILED',
        }

    ppe_model = _get_ppe_model()

    if ppe_model is None:
        return image, {
            'helmet': False,
            'gloves': False,
            'jacket': False,
            'shoes': False,
            'crack': False,
            'helmet_missing': False,
            'supported_ppe': {
                'helmet': False,
                'gloves': False,
                'jacket': False,
                'shoes': False,
                'no_helmet': False,
            },
            'status': 'MODEL FILES MISSING',
        }

    # Run inference on the frame
    ppe_results = ppe_model(image)
    ppe_labels = _labels_from_results(ppe_results, ppe_model)
    supported_ppe = _ppe_support_map(ppe_model)

    helmet = 'helmet' in ppe_labels
    gloves = 'gloves' in ppe_labels
    jacket = bool(set(ppe_labels).intersection({
        'vest',
        'safety_vest',
        'jacket',
        'reflective_vest',
    }))
    shoes = bool(set(ppe_labels).intersection({
        'boots',
        'shoes',
        'safety_boots',
    }))

    helmet_missing = 'no_helmet' in ppe_labels
    crack = False

    required_ppe_found = helmet and jacket and not helmet_missing
    status = "ALLOWED FOR MINING" if required_ppe_found else "NOT ALLOWED FOR MINING"

    annotated_frame = ppe_results[0].plot() if ppe_results else image

    return annotated_frame, {
        "helmet": helmet,
        "gloves": gloves,
        "jacket": jacket,
        "shoes": shoes,
        "crack": crack,
        "helmet_missing": helmet_missing,
        "supported_ppe": supported_ppe,
        "ppe_labels": ppe_labels,
        "status": status,
    }
