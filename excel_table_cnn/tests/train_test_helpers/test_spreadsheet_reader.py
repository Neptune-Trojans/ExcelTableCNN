import unittest
import torch
import numpy as np

from excel_table_cnn.dl_classification.spreadsheet_dataset import SpreadsheetDataset
from excel_table_cnn.train_test_helpers.utils import get_device


class TestResizeWithRowColCopy(unittest.TestCase):

    def setUp(self):
        self.c = 17
        self.device = get_device()
        self.processor = SpreadsheetDataset([self.make_matrix(11, 110)],[self.make_matrix(12, 110)], self.device)

    def make_matrix(self, h, w):
        return torch.arange(h * w * self.c, dtype=torch.float32, device=self.device).reshape(h, w, self.c)

    def test_no_resize(self):
        mat = self.make_matrix(4, 4)
        out = self.processor.resize_with_row_col_copy(mat, 4, 4)
        self.assertTrue(torch.equal(mat, out))

    def test_resize_rows_only(self):
        mat = self.make_matrix(2, 3)
        out = self.processor.resize_with_row_col_copy(mat, 5, 3)
        self.assertEqual(out.shape, (5, 3, self.c))
        self.assertTrue(torch.equal(out[2:], out[1:2].repeat(3, 1, 1)))

    def test_resize_cols_only(self):
        mat = self.make_matrix(3, 2)
        out = self.processor.resize_with_row_col_copy(mat, 3, 4)
        self.assertEqual(out.shape, (3, 4, self.c))
        self.assertTrue(torch.equal(out[:, 2:], out[:, 1:2].repeat(1, 2, 1)))

    def test_resize_both(self):
        mat = self.make_matrix(2, 2)
        out = self.processor.resize_with_row_col_copy(mat, 4, 5)
        self.assertEqual(out.shape, (4, 5, self.c))
        self.assertTrue(torch.equal(out[2:, :2, :], out[1:2, :2, :].repeat(2, 1, 1)))
        self.assertTrue(torch.equal(out[:, 2:, :], out[:, 1:2, :].repeat(1, 3, 1)))

    def test_shrink_should_trim(self):
        mat = self.make_matrix(5, 5)
        out = self.processor.resize_with_row_col_copy(mat, 3, 3)
        self.assertEqual(out.shape, (3, 3, self.c))
        self.assertTrue(torch.equal(out, mat[:3, :3, :]))

    def test_channel_mismatch_should_fail(self):
        mat = torch.zeros(3, 3, 16)
        with self.assertRaises(AssertionError):
            self.processor.resize_with_row_col_copy(mat, 4, 4)

    def test_zero_column_input(self):
        mat = torch.zeros(3, 0, self.c)  # width is 0!
        out = self.processor.resize_with_row_col_copy(mat, 5, 2)
        self.assertEqual(out.shape, (5, 2, self.c))


if __name__ == "__main__":
    unittest.main()
