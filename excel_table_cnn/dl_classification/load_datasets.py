import ast

import pandas as pd
import yaml

from excel_table_cnn.train_test_helpers.spreadsheet_reader import SpreadsheetReader


class DatasetManager:
    def __init__(self, config_path: str):
        """
        Initialize the DatasetManager with a path to the YAML config file.
        Loads dataset configurations into memory.
        """
        self.config_path = config_path
        self.datasets = self._load_config()
        self._spreadsheet_reader = SpreadsheetReader(640, 640)

    def _load_config(self) -> dict:
        """
        Load and parse the YAML configuration file.
        Returns a dictionary of datasets.
        """
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get("datasets", {})

    def load_datasets(self):
        tables_data = []
        backgrounds_data = []
        for name, settings in self.datasets.items():
            print(f'loading dataset: {name}')
            labels_file = settings["meta_data"]
            data_folder = settings["data_folder"]

            labels_df = pd.read_csv(labels_file)
            labels_df['table_region'] = labels_df['table_region'].apply(ast.literal_eval)

            tables, backgrounds = self._spreadsheet_reader.load_dataset_maps(labels_df, data_folder)
            tables_data.extend(tables)
            backgrounds_data.extend(backgrounds)

        return tables_data, backgrounds_data