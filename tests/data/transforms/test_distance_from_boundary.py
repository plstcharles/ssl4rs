import unittest
import numpy as np
import torch
import matplotlib.pyplot as plt

from ssl4rs.ssl4rs.data.transforms.distance_from_boundary import generate_boundary_distance_mask

class TestGenerateBoundaryDistanceMask(unittest.TestCase):

    def visualize_result(self, result, title):
        plt.imshow(result, cmap='hot', interpolation='nearest')
        plt.title(title)
        plt.colorbar()
        plt.show()

    def test_empty_class_label_map(self):
        class_label_map = np.zeros((5, 5), dtype=np.int32)
        target_class_label = 1
        expected_output = np.zeros((5, 5), dtype=np.int64)
        output = generate_boundary_distance_mask(class_label_map, target_class_label)
        np.testing.assert_array_equal(output, expected_output)
        # self.visualize_result(output, 'Empty Class Label Map')

    def test_single_target_class_label(self):
        class_label_map = np.array([
            [2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 1, 1, 1, 1, 1, 1, 1, 2],
            [2, 1, 0, 0, 0, 0, 0, 1, 2],
            [2, 1, 0, 0, 0, 0, 0, 1, 2],
            [2, 1, 0, 0, 0, 0, 0, 1, 2],
            [2, 1, 0, 0, 0, 0, 0, 1, 2],
            [2, 1, 1, 1, 1, 1, 1, 1, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2]
        ], dtype=np.int32)
        target_class_label = 1
        expected_output = np.zeros_like(class_label_map)
        output = generate_boundary_distance_mask(class_label_map, target_class_label, ignore_index=2)
        self.visualize_result(output, 'Single Target Class Label')

        # np.testing.assert_array_equal(output, expected_output)


    def test_three_target_class_label(self):
        class_label_map = np.array([
            [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
            [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
            [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
            [2, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 2],
            [2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 1, 2],
            [2, 1, 1, 1, 1, 1, 2, 2, 1, 0, 0, 0, 1, 2],
            [2, 1, 0, 0, 0, 1, 1, 2, 1, 0, 0, 0, 1, 2],
            [2, 1, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 1, 2],
            [2, 1, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 1, 2],
            [2, 1, 0, 0, 0, 1, 1, 2, 1, 0, 0, 0, 1, 2],
            [2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],

        ], dtype=np.int32)
        target_class_label = 1
        expected_output = np.zeros_like(class_label_map)
        output = generate_boundary_distance_mask(class_label_map, target_class_label, ignore_index=2)
        # self.visualize_result(output, 'Single Target Class Label')

        # np.testing.assert_array_equal(output, expected_output)


    def test_multiple_target_class_label(self):
        class_label_map = np.array([
            [2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 1, 1, 1, 1, 1, 1, 1, 2],
            [2, 1, 0, 0, 0, 0, 0, 1, 2],
            [2, 1, 0, 0, 0, 0, 0, 1, 2],
            [2, 1, 0, 0, 0, 0, 0, 1, 2],
            [2, 1, 0, 0, 0, 0, 0, 1, 2],
            [2, 1, 0, 0, 1, 1, 0, 1, 2],
            [2, 1, 1, 1, 2, 2, 1, 1, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 1, 1, 1, 2, 1, 1, 1, 2],
            [2, 1, 0, 1, 2, 1, 0, 1, 2],
            [2, 1, 0, 1, 2, 1, 0, 1, 2],
            [2, 1, 0, 1, 1, 1, 0, 1, 2],
            [2, 1, 0, 0, 0, 0, 0, 1, 2],
            [2, 1, 0, 0, 0, 0, 0, 1, 2],
            [2, 1, 1, 1, 1, 1, 1, 1, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2]
        ], dtype=np.int32)
        target_class_label = 1
        expected_output = np.zeros_like(class_label_map)
        output = generate_boundary_distance_mask(class_label_map, target_class_label, ignore_index=2)
        # self.visualize_result(output, 'Single Target Class Label')
        # np.testing.assert_array_equal(output, expected_output)

    def test_ignore_index(self):
        class_label_map = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 2, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.int32)
        target_class_label = 1
        ignore_index = 2
        expected_output = np.zeros_like(class_label_map)
        output = generate_boundary_distance_mask(class_label_map, target_class_label, ignore_index=ignore_index)
        # np.testing.assert_array_equal(output, expected_output)
        # self.visualize_result(output, 'Ignore Index')

    def test_torch_tensor_input(self):
        class_label_map = torch.tensor([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=torch.int32)
        target_class_label = 1
        expected_output = np.zeros_like(class_label_map)
        output = generate_boundary_distance_mask(class_label_map, target_class_label)
        # np.testing.assert_array_equal(output, expected_output)
        # self.visualize_result(output, 'Torch Tensor Input')

    def test_no_target_class_label(self):
        class_label_map = np.zeros((5, 5), dtype=np.int64)
        target_class_label = 1
        expected_output = np.zeros((5, 5), dtype=np.int64)
        output = generate_boundary_distance_mask(class_label_map, target_class_label)
        # np.testing.assert_array_equal(output, expected_output)
        # self.visualize_result(output, 'No Target Class Label')

if __name__ == '__main__':
    unittest.main()
