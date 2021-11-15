from typing import List
import numpy as np

class Transforms:
    """
    Transforms is a thin wrapper for applying multiple image transformations.

    Args
    ----
    transforms (List) : A list of image transformations that need to be applied on a matrix.

    Returns
    -------
    Augmented `np.ndarray` object.

    Example
    -------
    ```python
    t = Transforms([
            Normalize(mean=(0.5,0.5,0.5),std=(1,1,1)),
            RandomHorizontalFlip(p=0.5),
        ])
    matrix = t.transform(matrix)
    ```
    """
    def __init__(self, transforms : List) -> None:
        self.transforms = transforms

    def transform(self, matrix : np.ndarray) -> np.ndarray:
        for transformation in self.transforms:
            matrix = transformation.transform(matrix)
        return matrix