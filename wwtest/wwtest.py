import numpy as np
from scipy.stats import norm

class WWTestResult:
    def __init__(self, statistic, parameter, p_value, method, alternative, estimate):
        self.statistic = statistic
        self.parameter = parameter
        self.p_value = p_value
        self.method = method
        self.alternative = alternative
        self.estimate = estimate

    def __str__(self):
        return (f"Wilcoxon--Wigner Homogeneity Test\n"
                f"Test Statistic: {self.statistic:.5f}\n"
                f"Matrix Size: {self.parameter}\n"
                f"p-value: {self.p_value:.5g}\n"
                f"Alternative Hypothesis: {self.alternative}\n"
                f"Largest Eigenvalue: {self.estimate}")


def wwtest(data_matrix):
    # 1. Check if the input is a square matrix and contains no NaN values
    if not isinstance(data_matrix, np.ndarray) or data_matrix.shape[0] != data_matrix.shape[1]:
        raise ValueError("The input must be a square matrix.")
    if np.isnan(data_matrix).any():
        raise ValueError("The input matrix contains NaN values.")

    # 2. Check if the matrix is symmetric
    if not np.allclose(data_matrix, data_matrix.T):
        raise ValueError("The input must be a symmetric matrix.")

    n = data_matrix.shape[0]
    N = n * (n - 1) / 2

    # 3. Transform the upper triangular entries to ranks
    upper_tri_indices = np.triu_indices_from(data_matrix, k=1)
    upper_tri_values = data_matrix[upper_tri_indices]
    ranked_values = np.argsort(np.argsort(upper_tri_values)) + 1
    transformed_matrix = np.zeros_like(data_matrix, dtype=float)
    transformed_matrix[upper_tri_indices] = ranked_values / (N + 1)
    transformed_matrix += transformed_matrix.T

    # 4. Compute the first eigenvalue
    eigenvalues = np.linalg.eigvalsh(transformed_matrix)
    first_eigenvalue = eigenvalues[-1]

    # 5. Construct the test statistic
    sigma2 = 1 / 12 - 1 / (6 * (N + 1))
    mu1 = 0.5 * (n - 1) + 2 * sigma2
    tilde_sigma2 = 8 * sigma2**2 / n

    test_statistic = (first_eigenvalue - mu1) / np.sqrt(tilde_sigma2)

    # 6. Calculate the p-value
    p_value = 2 * norm.sf(np.abs(test_statistic))

    # 7. Return the test result
    result = WWTestResult(
        statistic=test_statistic,
        parameter=n,
        p_value=p_value,
        method="Wilcoxon--Wigner Homogeneity Test",
        alternative="The entries in the matrix are not identically distributed",
        estimate=first_eigenvalue,
    )

    return result

# Example usage
if __name__ == "__main__":
    mat = np.array([[1, 2, 3],
                    [2, 1, 1],
                    [3, 1, 1]])
    result = wwtest(mat)
    print(result)
