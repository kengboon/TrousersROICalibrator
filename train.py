from datetime import datetime
import os, gc

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
from tqdm import tqdm

from keypt_det.datasets import DeepFashion2Dataset, collate_fn
from keypt_det.utils import EarlyStopping

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

def train():
    start_time = datetime.now()

    # Filter category
    category_ids = [8]

    train_dataset = DeepFashion2Dataset(
        "datasets/deepfashion2/train-small",
        category_ids=category_ids,
        exclude_occulded=True
    )
    val_dataset = DeepFashion2Dataset(
        "datasets/deepfashion2/val-small",
        category_ids=category_ids,
        exclude_occulded=True
    )

    print(f"Data loaded, time taken={(datetime.now()-start_time).total_seconds():.2f}s")

    batch_size = 2
    grad_accum_batch = 2
    total_epochs = 100

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    num_classes = None if category_ids is None else len(category_ids) + 1 # 0 = background
    num_keypoints = 294  # Maximum keypoints for DeepFashion2 dataset
    model = get_model(num_classes=num_classes, num_keypoints=num_keypoints).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=0.001,
        total_iters=int(len(train_dataloader) / grad_accum_batch)
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.1,
        patience=5,
        threshold=0.01,
        threshold_mode="rel",
        cooldown=2,
        min_lr=1e-6
    )
    earlystop_checker = EarlyStopping(patience=10, min_delta=0.005, mode="min")

    train_losses = []
    val_losses = []
    earlystop_triggered = None
    for epoch in range(total_epochs):
        if earlystop_triggered:
            print("Early stopped.")
            break

        for phase in ("train", "val"):
            if phase == "train":
                dataloader = train_dataloader
                model.train()
                optimizer.zero_grad() # Reset gradients
            else:
                dataloader = val_dataloader
                #model.eval() # Do not change model for TorchVision's implementation

            epoch_losses = []
            accum_batch = 0
            with tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs} - {phase}") as pbar:
                for batch_i, (images, targets) in enumerate(dataloader):
                    # Move data to device
                    images = [image.to(device) for image in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    with torch.set_grad_enabled(phase == "train"):
                        loss_dict = model(images, targets)

                    # Combine losses
                    loss = sum(loss for loss in loss_dict.values())
                    epoch_losses.append(loss.item())

                    if phase == "train":
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(params, max_norm=10.0) # Gradient clipping
                        accum_batch += 1
                        if (accum_batch >= grad_accum_batch) or (batch_i == len(dataloader) - 1):
                            optimizer.step()
                            if epoch == 0:
                                warmup_scheduler.step() # Stepup LR for 1st epoch
                            accum_batch = 0
                            optimizer.zero_grad()
                    pbar.update()

            epoch_loss = np.array(epoch_losses).mean().item()
            if phase == "train":
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)
                if epoch > 0:
                    lr_scheduler.step(epoch_loss)
                earlystop_triggered = earlystop_checker.check(epoch_loss)
                improving = not earlystop_checker.is_stagnant
                if improving or earlystop_triggered or epoch == 0:
                    # Save checkpoint
                    ckpt = {
                        "epoch"             : epoch,
                        "model_weight"      : model.state_dict(),
                        "num_classes"       : num_classes,
                        "num_keypoints"     : num_keypoints,
                        "optimizer"         : optimizer.state_dict(),
                        "warmup"            : warmup_scheduler.state_dict(),
                        "lr_scheduler"      : lr_scheduler.state_dict(),
                        "earlystopping"     : earlystop_checker.state_dict(),
                        "train_losses"      : train_losses,
                        "val_losses"        : val_losses
                    }
                    ckpt_path = f"checkpoints/{start_time.strftime('%Y-%m-%d')}/{start_time.strftime('%H-%M-%S')}/epoch-{epoch+1}.pt"
                    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                    torch.save(ckpt, ckpt_path)

            torch.cuda.empty_cache()
            gc.collect()

        print(f"Train loss: {train_losses[epoch]:.4f}, val loss: {val_losses[epoch]:.4f}")

    # Visualize losses
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train()