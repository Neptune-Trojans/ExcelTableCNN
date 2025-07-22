import argparse
import ast
import pandas as pd

from excel_table_cnn.train_test_helpers.spreadsheet_reader import SpreadsheetReader
from runners.train_eval_runner import init_dataframe_view


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training configuration')
    parser.add_argument('--annotations_csv_file', type=str, help='annotations csv file')
    parser.add_argument('--spreadsheets_folder', type=str, help='preprocessed files path')
    parser.add_argument('--output_folder', type=str, help='preprocessed files path')

    args = parser.parse_args()

    init_dataframe_view()
    labels_df = pd.read_csv(args.annotations_csv_file)
    labels_df['table_region'] = labels_df['table_region'].apply(ast.literal_eval)

    spreadsheet_reader = SpreadsheetReader(300,300, args.output_folder)
    spreadsheet_reader.load_dataset_maps(labels_df, args.spreadsheets_folder)
