import ast

import pandas as pd

from excel_table_cnn.train_test_helpers.spreadsheet_reader import SpreadsheetReader
from runners.train_eval_runner import init_dataframe_view

def first_load_data():
    init_dataframe_view()
    dfs = pd.read_excel('/Users/arito/Data/anotation_data_outsoursing/RICOHERMOSO_table_labels_6-10.xlsx', sheet_name=None)
    result = []
    for sheet_name, df in dfs.items():
        # Apply prefix to file_path
        df["file_path"] = f"{sheet_name}/" + df["file_path"]

        result.append(df)

    result_df = pd.concat(result)
    result_df.to_csv('RICOHERMOSO_table_labels_6-10.csv', index=False)

if __name__ == '__main__':

    labels_df = pd.read_csv('/Users/arito/Data/anotation_data_outsoursing/RICOHERMOSO_table_labels_6-10.csv')
    labels_df['table_region'] = labels_df['table_region'].apply(ast.literal_eval)


    data_folder = '/Users/arito/.arito/Applications/excel_files_scraping'

    spreadsheet_reader = SpreadsheetReader(640,640)
    tables, backgrounds = spreadsheet_reader.load_dataset_maps(labels_df, data_folder)