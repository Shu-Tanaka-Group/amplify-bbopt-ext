import unittest

import numpy as np

from amplify_bbopt_ext.training import LatestDataset


def _make_data(n: int, num_variables: int = 3) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(n * num_variables, dtype=float).reshape(n, num_variables)
    y = np.arange(n, dtype=float)
    return x, y


class LatestDatasetInitTest(unittest.TestCase):
    def test_keeps_all_data_within_limit(self) -> None:
        x, y = _make_data(3)
        dataset = LatestDataset(x, y, size_limit=5)

        np.testing.assert_array_equal(dataset.x, x)
        np.testing.assert_array_equal(dataset.y, y)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.size_limit, 5)

    def test_truncates_to_latest_data(self) -> None:
        x, y = _make_data(5)
        dataset = LatestDataset(x, y, size_limit=2)

        np.testing.assert_array_equal(dataset.x, x[-2:])
        np.testing.assert_array_equal(dataset.y, y[-2:])
        self.assertEqual(len(dataset), 2)

    def test_default_limit_is_initial_size(self) -> None:
        x, y = _make_data(4)
        dataset = LatestDataset(x, y)

        self.assertEqual(dataset.size_limit, 4)
        np.testing.assert_array_equal(dataset.x, x)
        np.testing.assert_array_equal(dataset.y, y)

    def test_non_positive_limit_raises(self) -> None:
        x, y = _make_data(3)
        for size_limit in (0, -1):
            with self.assertRaises(ValueError):
                LatestDataset(x, y, size_limit=size_limit)

    def test_shape_mismatch_raises(self) -> None:
        x, _ = _make_data(3)
        with self.assertRaises(ValueError):
            LatestDataset(x, np.arange(2, dtype=float), size_limit=5)
        with self.assertRaises(ValueError):
            LatestDataset(x[0], np.arange(1, dtype=float), size_limit=5)


class LatestDatasetEmptyTest(unittest.TestCase):
    def test_empty_has_given_shape_and_limit(self) -> None:
        dataset = LatestDataset.empty(num_variables=4, size_limit=3)

        self.assertIsInstance(dataset, LatestDataset)
        self.assertEqual(dataset.x.shape, (0, 4))
        self.assertEqual(dataset.y.shape, (0,))
        self.assertEqual(len(dataset), 0)
        self.assertEqual(dataset.size_limit, 3)  # pyright: ignore[reportAttributeAccessIssue]

    def test_empty_with_non_positive_limit_raises(self) -> None:
        with self.assertRaises(ValueError):
            LatestDataset.empty(num_variables=4, size_limit=0)


class LatestDatasetAppendTest(unittest.TestCase):
    def test_append_within_limit(self) -> None:
        x, y = _make_data(2)
        dataset = LatestDataset(x, y, size_limit=3)
        new_x = np.array([10.0, 11.0, 12.0])
        dataset.append(new_x, 9.0)

        np.testing.assert_array_equal(dataset.x, np.vstack([x, new_x]))
        np.testing.assert_array_equal(dataset.y, [0.0, 1.0, 9.0])

    def test_append_drops_oldest(self) -> None:
        x, y = _make_data(3)
        dataset = LatestDataset(x, y, size_limit=3)
        new_x = np.array([10.0, 11.0, 12.0])
        dataset.append(new_x, 9.0)

        np.testing.assert_array_equal(dataset.x, np.vstack([x[1:], new_x]))
        np.testing.assert_array_equal(dataset.y, [1.0, 2.0, 9.0])
        self.assertEqual(len(dataset), 3)

    def test_append_multiple_rows_keeps_latest(self) -> None:
        x, y = _make_data(2)
        dataset = LatestDataset(x, y, size_limit=3)
        new_x, new_y = _make_data(4)
        new_x += 100.0
        new_y += 100.0
        dataset.append(new_x, new_y)

        np.testing.assert_array_equal(dataset.x, new_x[-3:])
        np.testing.assert_array_equal(dataset.y, new_y[-3:])

    def test_repeated_append_from_empty(self) -> None:
        dataset = LatestDataset.empty(num_variables=2, size_limit=2)
        for i in range(5):
            dataset.append(np.array([i, -i], dtype=float), float(i))
            self.assertEqual(len(dataset), min(i + 1, 2))

        np.testing.assert_array_equal(dataset.x, [[3.0, -3.0], [4.0, -4.0]])
        np.testing.assert_array_equal(dataset.y, [3.0, 4.0])

    def test_append_shape_mismatch_raises(self) -> None:
        x, y = _make_data(2)
        dataset = LatestDataset(x, y, size_limit=5)
        with self.assertRaises(ValueError):
            dataset.append(np.zeros((2, 3)), 1.0)


if __name__ == "__main__":
    unittest.main()
