import torch
from torch.utils.data import Dataset
import random


class SpreadsheetDataset(Dataset):
    def __init__(self, tables: list,  backgrounds:list, device):
        self._device = device

        self._epoch_iterations = 1000

        self._tables = [torch.tensor(table, dtype=torch.float32, device=self._device) for table in tables]
        self._backgrounds = [torch.tensor(background, dtype=torch.float32, device=self._device) for background in backgrounds]

        self._h_max = max(tensor.shape[0] for tensor in self._tables)
        self._w_max = max(tensor.shape[1] for tensor in self._tables)

    def __len__(self):
        # The length of the dataset is the number of spreadsheets
        return self._epoch_iterations

    def tile_matrix_randomly(self, background, max_tiles=10, max_attempts=50):
        H, W, _ = background.shape

        # Initialize ID map to track which tile occupies each location
        id_map = torch.zeros(H, W, dtype=torch.long, device=self._device)

        locations = []
        attempts = 0
        tile_id = 1  # Start tile IDs from 1

        while len(locations) < max_tiles and attempts < max_attempts:
            tile_idx = random.randint(0, len(self._tables) - 1)
            tile = self._tables[tile_idx]
            h, w, _ = tile.shape
            # Prevent tiles smaller than 2x2
            min_tile_size = 2
            h1 = random.randint(max(h, min_tile_size), H)
            w1 = random.randint(max(w, min_tile_size), W)
            try:
                new_tile = self.resize_with_row_col_copy(tile, h1, w1)
            except Exception as e:
                print(tile.shape)
                print(h1)
                print(w1)
                raise ValueError(f"Invalid range {e}")

            y1 = random.randint(0, H - h1)
            x1 = random.randint(0, W - w1)
            y2 = y1 + h1
            x2 = x1 + w1

            # Check if area is untiled (all zeros)
            if torch.all(id_map[y1:y2, x1:x2] == 0):
                background[y1:y2, x1:x2, :] = new_tile
                locations.append((x1, y1, x2 - 1, y2 - 1))
                #id_map[y1:y2, x1:x2] = tile_id  # Assign tile ID
                self.assign_tile_with_border(id_map, y1, y2, x1, x2, tile_id)
                tile_id += 1

            attempts += 1

        return locations

    def assign_tile_with_border(self, id_map, y1, y2, x1, x2, tile_id):
        """
        Assigns a tile ID to a rectangular region in a 2D tensor and sets a -1 border around it.

        Args:
            id_map (torch.Tensor): A 2D tensor of shape (H, W) where the assignment is made.
            y1, y2 (int): Start and end (exclusive) indices along height for the tile region.
            x1, x2 (int): Start and end (exclusive) indices along width for the tile region.
            tile_id (int): The ID to assign to the tile region.
        """
        H, W = id_map.shape

        # Compute padded border region (1-cell around the tile)
        y1_pad = max(y1 - 1, 0)
        y2_pad = min(y2 + 1, H)
        x1_pad = max(x1 - 1, 0)
        x2_pad = min(x2 + 1, W)

        # Set border region to -1
        id_map[y1_pad:y2_pad, x1_pad:x2_pad] = -1

        # Re-assign the tile_id in the core region
        id_map[y1:y2, x1:x2] = tile_id

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

        background_idx = random.randint(0, len(self._backgrounds) - 1)

        background_map = self._backgrounds[background_idx].clone()

        locations = self.tile_matrix_randomly(background_map)
        box_classes = [1]* len(locations)

        labels = {'boxes': torch.tensor(locations, dtype=torch.float32, device=self._device),
                  'labels': torch.tensor(box_classes, dtype=torch.int64, device=self._device)}

        #tensor = self.tensors.hwc_tensors[idx]
        # Permute tensor to C x H x W
        tensor = background_map.permute(2, 0, 1)

        return tensor, labels
