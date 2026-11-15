import unittest
from clock_executor import quote, PREFIX, SUFFIX


class ExecutorTests(unittest.TestCase):
    def test_quotes_are_data(self):
        self.assertEqual(quote('a[3]/$pin'), '"a\\[3\\]/\\$pin"')
        with self.assertRaises(ValueError):
            quote('a\nexit')

    def test_originals_fixed_and_restored(self):
        self.assertIn('setPlacementStatus FIRM', PREFIX)
        self.assertIn('original cell moved:', SUFFIX)
        self.assertIn('setPlacementStatus $status', SUFFIX)
        self.assertIn('check_placement -verbose', SUFFIX)
        self.assertIn('invalid PG', SUFFIX)


if __name__ == '__main__':
    unittest.main()
