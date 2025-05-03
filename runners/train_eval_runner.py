import argparse
import os
from datetime import datetime

import pandas as pd
import torch
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from excel_table_cnn.dl_classification.load_datasets import DatasetManager
from excel_table_cnn.dl_classification.model.train_eval import get_model, train_model, evaluate_model
from excel_table_cnn.dl_classification.spreadsheet_dataset import SpreadsheetDataset

data_folder_path = 'data'

def init_dataframe_view():
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

def collate_fn(batch):
    return tuple(zip(*batch))


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do all the things')
    parser.add_argument('--dataset_yaml', type=str, help='dataset yaml')


    parser.add_argument('--output_folder', type=str, help='output folder')
    parser.add_argument('--epochs_number', type=int, help='number of epochs')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='batch size')

    run_name = f"run-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    args = parser.parse_args()

    init_dataframe_view()

    image_height, image_width = (300, 300)

    model = get_model(17, image_height, image_width)

    dataset_manager = DatasetManager(args.dataset_yaml, image_height, image_width)
    tables, backgrounds = dataset_manager.load_datasets()

    device = get_device()

    train_dataset = SpreadsheetDataset(tables, backgrounds, device)
    test_dataset = SpreadsheetDataset(tables, backgrounds, device)


    batch_size = args.batch_size  # For different-sized inputs
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0001)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 40], gamma=0.1)

    wandb.init(
        project="synthetic-table-detector",
        name=run_name,  # optional
        config={
            "learning_rate": args.learning_rate,
            "epochs": args.epochs_number,
            "batch_size": batch_size,
        }
    )
    train_model(model, train_loader, optimizer,scheduler, args.epochs_number, device)

    torch.save(model.state_dict(), os.path.join(args.output_folder, 'weights.pt'))
    evaluate_model(model, test_loader, device)

    wandb.finish()
