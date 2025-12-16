import os

from safetensors.torch import save_model, load_model
import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_model(
        num_classes: int | None = None,
        num_keypoints: int | None = None
    ):
    # Load a pre-trained Keypoint RCNN model
    weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    model = keypointrcnn_resnet50_fpn(weights=weights)

    # Modify the model's head
    if num_classes is not None:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    if num_keypoints is not None:
        in_features = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
        model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(in_features, num_keypoints)

    return model

def load_model_from_checkpoint(path: str):
    ckpt = torch.load(path, map_location=device)
    num_classes = ckpt["num_classes"] if "num_classes" in ckpt else None
    num_keypoints = ckpt["num_keypoints"] if "num_keypoints" in ckpt else None
    model = get_model(num_classes=num_classes, num_keypoints=num_keypoints).to(device)
    if "model_weight" in ckpt:
        model.load_state_dict(ckpt["model_weight"])
    return model

if __name__ == "__main__":
    model = load_model_from_checkpoint("checkpoints/2025-12-12/15-07-04/epoch-48.pt")

    # Saving the model weight
    model_weight_path = "models/2025-12-12/15-07-04/epoch-48/model.safetensors"
    os.makedirs(os.path.dirname(model_weight_path), exist_ok=True)
    save_model(model, model_weight_path)

    # Try load the saved model weight
    load_model(model, model_weight_path, device=device)