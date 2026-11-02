import unittest
import pandas as pd
from model.datahandler import DataHandler, DataHandlerBIO, DataSplit, DummyDataHandler

# Run all tests in this file:
# python -m unittest test.datahandler_test


def make_df(ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame({'id': ids, 'text': [f'text {i}' for i in ids], 'labels': [i % 2 for i in ids]})


def make_split(ids: list[int]) -> DataSplit:
    return DataSplit(make_df(ids), {0: 'a', 1: 'b'}, tokenizer=None, max_len=16, multilabel=False)


def make_handler(df: pd.DataFrame = None) -> DummyDataHandler:
    """DataHandler without __init__, so no tokenizer has to be downloaded."""
    handler = DummyDataHandler.__new__(DummyDataHandler)
    handler.df = df
    handler.id2label = {0: 'a', 1: 'b'}
    handler.nr_classes = 2
    handler.is_multilabel = False
    handler.tokenizer = None
    handler.max_len = 16
    handler.train = handler.test = handler.val = None
    handler.train_size = 0.8
    handler.use_val = False
    return handler


class TestSplitOverlap(unittest.TestCase):

    def test_disjoint_splits_pass(self):
        DataHandler._check_no_overlap(
            make_split([1, 2, 3]), make_split([4, 5]), make_split([6, 7]))

    def test_train_test_overlap_fails(self):
        with self.assertRaisesRegex(ValueError, 'train and test'):
            DataHandler._check_no_overlap(make_split([1, 2, 3]), make_split([3, 4]))

    def test_train_val_overlap_fails(self):
        with self.assertRaisesRegex(ValueError, 'train and val'):
            DataHandler._check_no_overlap(
                make_split([1, 2, 3]), make_split([4, 5]), make_split([3, 6]))

    def test_val_test_overlap_fails(self):
        with self.assertRaisesRegex(ValueError, 'val and test'):
            DataHandler._check_no_overlap(
                make_split([1, 2, 3]), make_split([4, 5]), make_split([5, 6]))

    def test_reused_splits_with_val_equal_test_fail(self):
        # The original bug: val.csv identical to test.csv
        handler = make_handler()
        handler.train, handler.test, handler.val = make_df([1, 2, 3]), make_df([4, 5]), make_df([4, 5])
        handler.use_val = True
        with self.assertRaisesRegex(ValueError, 'val and test'):
            handler.get_strat_split(train_size=0.8, use_val=True)

    def test_new_split_with_val_is_disjoint(self):
        handler = make_handler(make_df(list(range(100))))
        train, test, val = handler.get_strat_split(train_size=0.6, use_val=True)
        train_ids, test_ids, val_ids = (set(s.df['id']) for s in (train, test, val))
        self.assertEqual((len(train_ids), len(val_ids), len(test_ids)), (60, 20, 20))
        self.assertFalse(val_ids & test_ids)
        self.assertEqual(train_ids | val_ids | test_ids, set(range(100)))

    def test_ner_val_test_overlap_fails(self):
        handler = DataHandlerBIO.__new__(DataHandlerBIO)
        handler.train, handler.test, handler.val = make_df([1, 2, 3]), make_df([4, 5]), make_df([5, 6])
        with self.assertRaisesRegex(ValueError, 'Overlap between splits'):
            handler.check_overlap()


if __name__ == '__main__':
    unittest.main()
