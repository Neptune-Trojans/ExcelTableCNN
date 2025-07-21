import ast
import pandas as pd

from excel_table_cnn.train_test_helpers.spreadsheet_reader import SpreadsheetReader
from runners.train_eval_runner import init_dataframe_view


if __name__ == '__main__':
    init_dataframe_view()
    labels_df = pd.read_csv('/Users/arito/Data/anotation_data_outsoursing/image_annotations_to_spreadsheet/table_labels.csv')
    labels_df['table_region'] = labels_df['table_region'].apply(ast.literal_eval)

    data_folder = '/Users/arito/Data/spreadsheets_pool'

    features_output_folder = '/Users/arito/.arito/Applications/excel_table_train'

    spreadsheet_reader = SpreadsheetReader(300,300, features_output_folder)
    spreadsheet_reader.load_dataset_maps(labels_df, data_folder)
