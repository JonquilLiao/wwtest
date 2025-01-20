# wwtest

**Wilcoxon--Wigner Homogeneity Test for Matrix Data**

This package provides a statistical test to determine whether the entries in a symmetric matrix are identically distributed. The test is based on the Wilcoxon--Wigner matrix.

## Installation

You can install the package directly from GitHub using:

```bash
pip install git+https://github.com/JonquilLiao/wwtest.git
```

## Usage

Here's how to use the `wwtest` package:

```python
import numpy as np
from wwtest import wwtest

# Create a symmetric matrix
mat = np.array([[1, 2, 3], [2, 1, 1], [3, 1, 1]])

# Run the Wilcoxon--Wigner homogeneity test
result = wwtest(mat, 'eigenvalue')
print(result)
```

