import unittest
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from src.manual_model_building.utilities import generate_polynomials


class TestPolynomialFeatures(unittest.TestCase):
    def setUp(self):
        # Test cases: each array represents a row, with varying degrees
        self.test_data = [np.array([1, 2, 7, 1]), np.array([3, 4, 5, 1])]
        self.degrees = [1, 2, 3]

    def test_single_row_polynomial_features(self):
        for x_i in self.test_data:
            for degree in self.degrees:
                # Expected output using sklearn's PolynomialFeatures
                poly = PolynomialFeatures(degree=degree, include_bias=True)
                expected_output = poly.fit_transform([x_i]).flatten()

                # Custom function output
                custom_output = generate_polynomials.generate_polynomial_features(
                    x_i, degree, add_bias=True
                )

                # Assert the two outputs are equal
                np.testing.assert_array_almost_equal(
                    custom_output,
                    expected_output,
                    err_msg=f"Failed for input {x_i} with degree {degree}",
                )

    def test_multi_row_polynomial_features(self):
        for degree in self.degrees:
            # Example matrix with multiple rows
            try:
                x_matrix = np.array(self.test_data)
            except ValueError as e:
                raise ValueError(
                    f"ValueError for degree: {degree}, original error: {e}"
                )

            # Expected output using sklearn's PolynomialFeatures
            poly = PolynomialFeatures(degree=degree, include_bias=True)
            expected_output = poly.fit_transform(x_matrix)

            # Custom function output
            custom_output = generate_polynomials.multi_row_polynomial_features(
                x_matrix, degree
            )

            # Assert the two outputs are equal
            np.testing.assert_array_almost_equal(
                custom_output,
                expected_output,
                err_msg=f"Failed for matrix {x_matrix} with degree {degree}",
            )


if __name__ == "__main__":
    unittest.main()
