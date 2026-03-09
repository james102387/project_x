"""Calculator tool — executes arithmetic operations via NumPy."""

import numpy as np


def execute_add(args: list[int | float]) -> int | float:
    """Sum a list of numbers."""
    return np.sum(args)


OPERATIONS = {
    "add": execute_add,
}
