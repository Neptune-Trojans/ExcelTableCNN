import torch
from torch.utils.data import Dataset
import random

from .tensors import DataframeTensors
from ..train_test_helpers import get_table_features


def get_bounding_box(table_ranges):
    boxes = torch.tensor(
        [
            [min_col, min_row, max_col, max_row]  # x_min, y_min, x_max, y_max
            for min_col, min_row, max_col, max_row in table_ranges
        ],
        dtype=torch.float32,
    )
    # Assuming '1' is the label for tables:
    labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
    return {"boxes": boxes, "labels": labels}


# class SpreadsheetDataset(Dataset):
#     def __init__(self, tensors: DataframeTensors):
#         self.tensors = tensors
#         self.num_cell_features = tensors.num_cell_features
#
#     def __len__(self):
#         # The length of the dataset is the number of spreadsheets
#         return len(self.tensors.hwc_tensors)
#
#     def __getitem__(self, idx):
#         tensor = self.tensors.hwc_tensors[idx]
#         # Permute tensor to C x H x W
#         tensor = tensor.permute(2, 0, 1)
#
#         # Get labels
#         labels = get_bounding_box(self.tensors.zero_indexed_table_ranges[idx])
#
#
#
#         return tensor, labels


class SpreadsheetDataset(Dataset):
    def __init__(self, tensors: DataframeTensors):
        # self.tensors = tensors
        # self.num_cell_features = tensors.num_cell_features

        self._epoch_iterations = 1000
        self._pairs = [self.generate_valid_pair() for _ in range(1000)]
        self.example_features = self.get_example_features()



    def get_example_features(self):
        features_df = get_table_features('../excel_table_cnn/dl_classification/2231.xlsx', 'Sheet1')
        features_df['file_path'] = '2231.xlsx'
        features_df['sheet_name'] ='Sheet1'
        features_df['table_range'] = [["A2:K7"]] * len(features_df)
        features_df = DataframeTensors(features_df)
        return features_df.hwc_tensors[0][1:]


    def __len__(self):
        # The length of the dataset is the number of spreadsheets
        return len(self._pairs)

    @staticmethod
    def generate_valid_pair(limit=30000):
        while True:
            a = random.randint(50, limit // 50)
            b = random.randint(50, limit // 50)
            if a * b < limit:
                return [a, b]

    def assign_matrix_randomly(self, spreadsheet_map, matrix):
        H, W, _ = spreadsheet_map.shape
        h, w, _ = matrix.shape

        if h > H or w > W:
            raise ValueError("Matrix is larger than the map in one or more dimensions.")

        # Choose a random top-left location (x1, y1)
        x1 = random.randint(0, H - h)
        y1 = random.randint(0, W - w)
        x2 = x1 + h
        y2 = y1 + w

        # Assign matrix to that location
        spreadsheet_map[x1:x2, y1:y2, :] = matrix

        return x1, y1, x2, y2

    def __getitem__(self, idx):

        H, W = self._pairs[idx]
        tensor = torch.zeros(H, W, 10)
        x1, y1, x2, y2 = self.assign_matrix_randomly(tensor, self.example_features)
        labels = {'boxes': torch.tensor([[ x1, y1, x2, y2]], dtype=torch.float32), 'labels': torch.tensor([1], dtype=torch.int64)}

        #tensor = self.tensors.hwc_tensors[idx]
        # Permute tensor to C x H x W
        tensor = tensor.permute(2, 0, 1)

        # Get labels
        #labels = get_bounding_box(self.tensors.zero_indexed_table_ranges[idx])



        return tensor, labels
