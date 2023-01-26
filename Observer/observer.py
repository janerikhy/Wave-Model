# observer.py
import numpy as np
from abc import ABC, abstractclassmethod

from utils import Rz, Smat, J, timeit, pipi


class Observer(ABC):
    '''
    Base class for DP observer
    '''

    def __init__(self):
        pass

    def integrate(self):
        pass

    