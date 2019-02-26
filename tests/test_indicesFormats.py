from unittest import TestCase

from nirmapper.model import IndicesFormat


class TestIndicesFormats(TestCase):

    def test_get_indices_formats_from_string(self):
        format_str = "T2F_N3F_V3F"
        expected = [IndicesFormat.T2F, IndicesFormat.N3F, IndicesFormat.V3F]

        self.assertEqual(IndicesFormat.get_indices_formats_from_string(format_str), expected)

    def test_get_indices_formats_from_single_string(self):
        format_str = "T2F"
        expected = [IndicesFormat.T2F]

        self.assertEqual(IndicesFormat.get_indices_formats_from_string(format_str), expected)

    def test_get_indices_formats_from_wrong_string(self):
        format_str = "T1F_N3F_V3F"

        with self.assertRaises(KeyError):
            IndicesFormat.get_indices_formats_from_string(format_str)
