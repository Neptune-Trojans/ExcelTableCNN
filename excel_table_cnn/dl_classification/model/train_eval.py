import torch
import wandb
from torch.utils.data import DataLoader
from collections import defaultdict
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from excel_table_cnn.dl_classification.model.detr_detector import DetrResNet18Mapped


def get_model(in_channels=2, max_height=200, max_width=200):
    #model = FasterRCNNMobileNetMapped2(input_channels=in_channels, num_classes=2, image_size=(max_height, max_width))
    model = DetrResNet18Mapped(num_classes=2)
    return model


def get_dataloader(dataset):
    def collate_fn(batch):
        return tuple(zip(*batch))

    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    return loader


def train_model(model, train_loader, optimizer, scheduler, num_epochs, device):

    model.to(device)

    # Set the model in training mode
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        loss_sums = defaultdict(float)
        for images, targets in train_loader:

            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Reset gradients
            optimizer.zero_grad()
            images = torch.stack(images, dim=0)

            # Forward pass
            result = model(images, targets)
            loss_dict = result.loss_dict
            losses = sum(loss for key, loss in loss_dict.items() if key.startswith('loss_'))

            # Accumulate epoch loss
            epoch_loss += losses.item()

            # Track all values (including cardinality_error) for logging/debugging
            for key, value in loss_dict.items():
                loss_sums[key] += value.item()

            # Backpropagation
            losses.backward()
            optimizer.step()

        # Step the scheduler at the end of each epoch
        scheduler.step()


        avg_loss_values_str = " | ".join([f"{name}: {value / len(train_loader):.4f}" for name, value in loss_sums.items()])
        avg_total = epoch_loss / len(train_loader)
        lr = scheduler.get_last_lr()[0]

        print(f"[Epoch {epoch}] {avg_loss_values_str} | total: {avg_total:.4f} | lr: {lr:.6f}")
        wandb.log({
            "epoch": epoch,
            "train/loss_total": avg_total,
            **{f"train/{k}": v / len(train_loader)  for k, v in loss_sums.items()},
            "lr": scheduler.get_last_lr()[0],
        })



def evaluate_model(model, test_loader, device, iou_threshold=0.5, conf_score=0.3):
    model.to(device)
    model.eval()

    all_detections = defaultdict(list)
    all_ground_truths = defaultdict(list)
    metric = MeanAveragePrecision()

    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            images = torch.stack(images, dim=0)
            preds = model(images)
            # Update it with your batches
            metric.update(preds, targets)

    results = metric.compute()

    wandb.log({f"val/{k}": v.item() if hasattr(v, 'item') else v for k, v in results.items()})
    print(results)


def get_model_output(model, test_loader, device):
    model.to(device)
    model.eval()  # Set the model in inference mode

    eval_loss = 0
    for images, targets in test_loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            # Forward pass
            outputs = model(images)
            print(outputs)
