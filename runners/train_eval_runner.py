import argparse
import ast
import os
from datetime import datetime

import pandas as pd
import torch
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader


from excel_table_cnn.dl_classification.model.train_eval import get_model, train_model, evaluate_model
from excel_table_cnn.dl_classification.spreadsheet_dataset import SpreadsheetDataset
from excel_table_cnn.train_test_helpers.spreadsheet_reader import SpreadsheetReader
from excel_table_cnn.train_test_helpers.utils import get_device

data_folder_path = 'data'

def init_dataframe_view():
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

def collate_fn(batch):
    return tuple(zip(*batch))





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training configuration')
    parser.add_argument('--annotations_csv_file', type=str, help='annotations csv file')
    parser.add_argument('--preprocessed_folder', type=str, help='preprocessed files path')

    parser.add_argument('--output_folder', type=str, help='output folder')

    parser.add_argument('--epochs_number', type=int, default=1, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='batch size')
    args = parser.parse_args()

    labels_df = pd.read_csv(args.annotations_csv_file)
    labels_df['table_region'] = labels_df['table_region'].apply(ast.literal_eval)

    run_name = f"run-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    train_df = labels_df.iloc[500:600].copy()
    val_df = labels_df.iloc[500:600].copy()


    init_dataframe_view()
    device = get_device()
    height, width = (300, 300)

    model = get_model(17, height, width)
    model = model.to(device=device)

    #dataset_manager = DatasetManager(args.dataset_yaml, image_height, image_width)
    #tables, backgrounds = dataset_manager.load_datasets()



    sp_reader = SpreadsheetReader(width, height, args.preprocessed_folder)

    train_dataset = SpreadsheetDataset(train_df, args.batch_size, sp_reader, device)
    test_dataset = SpreadsheetDataset(val_df, args.batch_size, sp_reader, device)



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
