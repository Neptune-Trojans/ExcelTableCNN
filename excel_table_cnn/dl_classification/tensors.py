from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from openpyxl.utils.cell import coordinate_to_tuple, range_boundaries


def parse_coordinate(coordinate):
    # Convert Excel-style coordinate (e.g., 'A1') to numerical indices
    # Placeholder for actual implementation
    row_index, col_index = coordinate_to_tuple(coordinate)
    return (row_index - 1, col_index - 1)


def parse_table_range(table_range):
    # Convert table range string to numerical coordinates
    # Placeholder for actual implementation
    min_col, min_row, max_col, max_row = range_boundaries(table_range)
    # Match the R-CNN convention: x_min, y_min, x_max, y_max
    return [min_col - 1, min_row - 1, max_col - 1, max_row - 1]


def preprocess_features(row):
    # Convert the pandas Series directly to a tensor
    return torch.tensor(row.astype(float).values, dtype=torch.float16)


class DataframeTensors:
    def __init__(self, dataframe):
        # Make tensors in CxHxW format
        self.hwc_tensors: List[torch.Tensor] = []
        self.zero_indexed_table_ranges: List[List[Tuple[int, int]]] = []
        self.num_cell_features: Optional[int] = None

        non_feature_columns = ["coordinate", "file_path", "sheet_name", "table_range"]

        # Group by file_path and sheet_name to process each sheet separately
        grouped = dataframe.groupby(["file_path", "sheet_name"])
        for _, group in tqdm(grouped, total=len(grouped), desc="Creating tensors and labels"):
            max_rows, max_cols = self._get_max_dimensions(group)
            if self.num_cell_features is None:
                self.num_cell_features = len(group.columns) - len(non_feature_columns)

            # sheet_tensor = torch.zeros((max_rows, max_cols, self.num_cell_features))
            #
            # for _, row in group.iterrows():
            #     row_idx, col_idx = parse_coordinate(row["coordinate"])
            #     cell_features = preprocess_features(row.drop(non_feature_columns))
            #     sheet_tensor[row_idx, col_idx, :] = cell_features
            sheet_tensor = self.process_sheet(group, max_rows, max_cols, self.num_cell_features, non_feature_columns)
            self.hwc_tensors.append(sheet_tensor)

            table_ranges = [
                parse_table_range(tr) for tr in group["table_range"].iloc[0]
            ]
            self.zero_indexed_table_ranges.append(table_ranges)

    def process_sheet(self, group, max_rows, max_cols, num_cell_features, non_feature_columns):
        sheet_tensor = torch.zeros((max_rows, max_cols, num_cell_features), dtype=torch.float32)

        # Extract coordinates
        coordinates = np.array([parse_coordinate(coord) for coord in group["coordinate"]])
        row_indices, col_indices = coordinates[:, 0], coordinates[:, 1]

        # Extract and process features
        feature_matrix = np.stack(group.drop(columns=non_feature_columns).apply(preprocess_features, axis=1).to_numpy())

        # Assign values efficiently
        sheet_tensor[row_indices, col_indices, :] = torch.tensor(feature_matrix, dtype=torch.float32)

        return sheet_tensor[:,:,17]

    def _get_max_dimensions(self, group):
        # Compute the max row and column indices for this spreadsheet
        # max_row, max_col = 0, 0
        # for _, row in group.iterrows():
        #     row_idx, col_idx = parse_coordinate(row["coordinate"])
        #     max_row = max(max_row, row_idx)
        #     max_col = max(max_col, col_idx)

        # Extract row and column indices using vectorized operations
        coordinates = group["coordinate"].apply(parse_coordinate)

        # Convert to a DataFrame for efficient max calculation
        coords_df = pd.DataFrame(coordinates.tolist(), columns=["row_idx", "col_idx"])

        # Get max row and column in one operation
        max_row, max_col = coords_df.max()
        return max_row + 1, max_col + 1  # Add 1 because indices are zero-based
