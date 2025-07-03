import unittest
import joblib
import numpy as np

class TestModel(unittest.TestCase):
    def test_prediction_shape(self):
        model = joblib.load("models/decision_tree_model.pkl")  # update path if needed
        sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # dummy sample
        prediction = model.predict(sample)
        self.assertEqual(len(prediction), 1)

if __name__ == '__main__':
    unittest.main()
