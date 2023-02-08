import math
from abc import ABC, abstractmethod
import numpy as np

class Constraint(ABC):
    """
    Abstract base class for a set
    of linear, convex constraints
    """

    def __init__(self):

        self._C, self._b, self._n = self._linearized_constraint()

    @abstractmethod
    def _linearized_constraint(self):
        """
        Abstract method, to be overridden.
        """

    @property
    def constraints(self):
        """
        Constraints in matrix and vector form
        """
        return self._C, self._b, self._n
