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
    def __init__(self, template_path: str):
        # self.tensors = tensors
        # self.num_cell_features = tensors.num_cell_features

        self._epoch_iterations = 1000
        self._pairs = [self.generate_valid_pair() for _ in range(1000)]
        self.example_features = self.get_example_features(template_path)



    def get_example_features(self, template_path):
        features_df = get_table_features(template_path, 'Sheet1')
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

    # def assign_matrix_randomly(self, spreadsheet_map, matrix):
    #     H, W, _ = spreadsheet_map.shape
    #     h, w, _ = matrix.shape
    #
    #     # Choose a random top-left location (x1, y1)
    #     y1 = random.randint(0, H - h)
    #     x1 = random.randint(0, W - w)
    #     x2 = x1 + w
    #     y2 = y1 + h
    #
    #     # Assign matrix to that location
    #     spreadsheet_map[y1:y2, x1:x2, :] = matrix
    #
    #     return x1, y1, x2, y2

    def tile_matrix_randomly(self, map_tensor, tile, max_tiles=10, max_attempts=30):
        H, W, C = map_tensor.shape

        n_tiles = random.randint(1, max_tiles)

        # Initialize ID map to track which tile occupies each location
        id_map = torch.zeros(H, W, dtype=torch.long)

        locations = []
        attempts = 0
        tile_id = 1  # Start tile IDs from 1

        while len(locations) < n_tiles and attempts < max_attempts:
            h, w, _ = tile.shape
            h1 = random.randint(h, H)
            w1 = random.randint(w, W)

            new_tile = self.resize_with_row_col_copy(tile, h1, w1)

            y1 = random.randint(0, H - h1)
            x1 = random.randint(0, W - w1)
            y2 = y1 + h1
            x2 = x1 + w1

            # Check if area is untiled (all zeros)
            if torch.all(id_map[y1:y2, x1:x2] == 0):
                map_tensor[y1:y2, x1:x2, :] = new_tile
                locations.append((x1, y1, x2, y2))
                id_map[y1:y2, x1:x2] = tile_id  # Assign tile ID
                tile_id += 1

            attempts += 1

        return locations

    def resize_with_row_col_copy(self, matrix, h1, w1):
        h, w, c = matrix.shape

        new_matrix = torch.zeros(h1, w1, 17)
        # Start with trimmed or same-size version
        new_matrix[:h, :w,  :] = matrix

        # If we need more rows
        if h1 > h:
            last_row = matrix[h - 1:h, :, :]  # shape: (1, w1, c)
            extra_rows = last_row.repeat(h1 - h, 1, 1)  # repeat rows
            new_matrix[h:h1, :w, :] = extra_rows

        # If we need more columns
        if w1 > w:
            last_col = new_matrix[:, w - 1:w, :]  # shape: (h1, 1, c)
            extra_cols = last_col.repeat(1, w1 - w, 1)  # repeat columns
            new_matrix[:, w:w1, :] = extra_cols

        return new_matrix

    def __getitem__(self, idx):

        H, W = self._pairs[idx]
        #tensor = torch.zeros(H, W, 10)
        tensor = torch.randint(0, 2, (H, W, 17), dtype=torch.float32)

        locations = self.tile_matrix_randomly(tensor, self.example_features)
        box_classes = [1]* len(locations)
        labels = {'boxes': torch.tensor(locations, dtype=torch.float32), 'labels': torch.tensor(box_classes, dtype=torch.int64)}

        #tensor = self.tensors.hwc_tensors[idx]
        # Permute tensor to C x H x W
        tensor = tensor.permute(2, 0, 1)

        # Get labels
        #labels = get_bounding_box(self.tensors.zero_indexed_table_ranges[idx])



        return tensor, labels
