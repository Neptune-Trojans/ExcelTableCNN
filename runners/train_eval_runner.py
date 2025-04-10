import ast
import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from excel_table_cnn.dl_classification.model.train_eval import get_model, train_model, evaluate_model
from excel_table_cnn.dl_classification.spreadsheet_dataset import SpreadsheetDataset
from excel_table_cnn.dl_classification.tensors import DataframeTensors
from excel_table_cnn.train_test_helpers.train_test_composer import get_train_test

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
    parser.add_argument('--labels_file', type=str, help='file with the labels')
    parser.add_argument('--data_folder', type=str, help='excel files folder')
    parser.add_argument('--output_folder', type=str, help='output folder')
    parser.add_argument('--epochs_number', type=int, help='number of epochs')
    args = parser.parse_args()


    init_dataframe_view()
    #train_df, test_df = get_train_test(args.labels_file, args.data_folder, args.output_folder)

    train_df = pd.read_pickle(os.path.join(args.output_folder, "train_features.pkl"))
    test_df = pd.read_pickle(os.path.join(args.output_folder, "test_features.pkl"))

    train_df['table_range'] = train_df['table_range'].apply(ast.literal_eval)
    test_df['table_range'] = test_df['table_range'].apply(ast.literal_eval)


    train_df = DataframeTensors(train_df)
    test_df = DataframeTensors(test_df)



    train_dataset = SpreadsheetDataset(train_df)
    test_dataset = SpreadsheetDataset(test_df)


    batch_size = 1  # For different-sized inputs
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    device = get_device()
    model = get_model(17)


    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    train_model(model, train_loader, optimizer, 10, device)


    evaluate_model(model, test_loader, device)
