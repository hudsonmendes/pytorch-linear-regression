import unittest
import pandas as pd

from src.logistic_regression import LogisticRegression

class TestLogisticRegression(unittest.TestCase):

    def setUp(self) -> None:
        self.data = pd.read_csv('./tests/fixtures/sample.csv')
        self.subject = LogisticRegression(1, 1)

    def test_train(self):
        # TODO: test training step
