import unittest
from data_processing import preprocess_text

class TestPreprocessing(unittest.TestCase):

    def test_preprocess_text(self):
        comment = "This is a sample comment with some Stop words and Punctuation!"
        expected_output = "sample comment stop words punctuation"
        self.assertEqual(preprocess_text(comment), expected_output)

if __name__ == '__main__':
    unittest.main()
