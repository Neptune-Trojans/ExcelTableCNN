import torch
from torch.utils.data import Dataset
import random


class SpreadsheetDataset(Dataset):
    def __init__(self, feature_maps: list, device):
        # self.tensors = tensors
        # self.num_cell_features = tensors.num_cell_features
        self._device = device

        self._epoch_iterations = 1000

        self._feature_maps = feature_maps
        self._feature_maps_shapes = [tensor.shape for tensor in feature_maps]
        self._h_max = max(tensor.shape[0] for tensor in feature_maps)
        self._w_max = max(tensor.shape[1] for tensor in feature_maps)

        self._pairs = [self._generate_valid_pairs() for _ in range(1000)]


    def __len__(self):
        # The length of the dataset is the number of spreadsheets
        return self._epoch_iterations


    def _generate_valid_pairs(self):
            H = self._h_max
            W = self._w_max

            return H, W

    def tile_matrix_randomly(self, map_tensor, max_tiles=10, max_attempts=50):
        H, W, C = map_tensor.shape


        # Initialize ID map to track which tile occupies each location
        id_map = torch.zeros(H, W, dtype=torch.long, device=self._device)

        locations = []
        attempts = 0
        tile_id = 1  # Start tile IDs from 1

        while len(locations) < max_tiles and attempts < max_attempts:
            tile_idx = random.randint(0, len(self._feature_maps) - 1)
            tile = self._feature_maps[tile_idx]
            h, w, _ = self._feature_maps_shapes[tile_idx]
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

        new_matrix = torch.zeros(h1, w1, 17, device=self._device)
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
        tensor = torch.zeros(H, W, 17, device=self._device)
        tensor[:, :, 0] = 1.0

        locations = self.tile_matrix_randomly(tensor)
        box_classes = [1]* len(locations)

        labels = {'boxes': torch.tensor(locations, dtype=torch.float32, device=self._device),
                  'labels': torch.tensor(box_classes, dtype=torch.int64, device=self._device)}

        #tensor = self.tensors.hwc_tensors[idx]
        # Permute tensor to C x H x W
        tensor = tensor.permute(2, 0, 1)

        return tensor, labels
