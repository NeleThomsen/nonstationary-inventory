import numpy as np
from abc import abstractmethod
from environments.envClass import Environment

def rand_argmax(arms) -> int:
    """
    Return argmax of a list breaking ties randomly
    :param arms: list of input values
    :return: position of a highest value in input list
    """
    arms[np.isnan(arms)] = 0
    maxValue = max(arms)
    index = [i for i in range(len(arms)) if arms[i] == maxValue]
    return int(np.random.choice(index))


class BaseAlgo:
    """
    Abstract class from which all algorithm classes that implement a specific algorithm are inherited.
    """
    def __int__(self, K: int, T: int, L: int = 0, model: str = "lost_sales", verbose: bool = False):
        self.K = K
        self.T = T
        self.L = L
        self.model = model
        self.verbose = verbose
        
    @abstractmethod
    def clear(self):
        """Resets member variables of algorithm class instance."""
        raise NotImplementedError("Method clear() has to be implemented in the class inherited from BaseAlgo.")

    @abstractmethod
    def selectAction(self, environment: Environment) -> int:
        """
        :return: selected action
        """
        raise NotImplementedError("Method selectAction() has to be implemented in the class inherited from BaseAlgo.")

    @abstractmethod
    def updateAlgo(self, arm: int, cost: float, sales: float | int, environment: Environment):
        """
        :param arm: the action taken (index of bslevels)
        :param cost: immediate cost incurred
        :param sales: quantity sold
        :param environment: environment instance
        """
        raise NotImplementedError("Method updateObservations() has to be implemented in class inherited from BaseAlgo.")

    def __repr__(self):
        return str(self.__class__.__name__)
