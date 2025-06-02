"""Learning rate scheduler classes."""

from typing import Callable, List

import numpy as np

from lib.optimizers import Optimizer


class LambdaLR:

    def __init__(self, optimizer: Optimizer, lr_lambda: Callable[[int], float]):
        """Sets the learning rate to the initial lr times a given function.

        Args:
            optimizer: The optimizer to wrap.
            lr_lambda: A function that takes the current epoch as an argument
                       and returns the corresponding learning rate.
        """
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.last_epoch = 0
        self.initial_lr = np.copy(optimizer.lr)
        self.lr_lambda = lr_lambda

    def step(self):
        """To be called after each epoch. Update optimizer.lr"""
        self.last_epoch += 1
        self.optimizer.lr = self.lr_lambda(self.last_epoch)


class PiecewiseConstantLR(LambdaLR):

    def __init__(self, optimizer: Optimizer, epochs: List[int], learning_rates: List[float]):
        """Set learning rate as piecewise constant steps.

        This class inherits from LambdaLR and implements the lambda
        function that maps the current epoch to the corresponding learning rate.

        Args:
            optimizer: The optimizer to wrap.
            epochs: List of epoch indices. Must be increasing. Determines at which epoch
                    the lr has to change.
            learning_rates: New learning rate for each epoch index. Determines what learning rate
                            is used in between the epoch intervals.
        """
        assert len(epochs) == len(learning_rates)
        self.epochs = epochs
        self.learning_rates = [optimizer.lr] + learning_rates
        super().__init__(optimizer, self._get_lambda())

    def _get_lambda(self) -> Callable[[int], float]:
        """Create the function to determine the learning rate given the epoch.

        Returns:
            Lambda LR function.
        """

        def lr_lambda(epoch: int) -> float:
            """Return learning rate given epoch.

            This function receives as argument the current epoch and, based on the list
            of epoch indices that determine when to change the learning rate, returns the
            correct learning rate value for the given epoch.

            Note:
                If you struggle with this exercise, feel free to look at the
                testcase in tests/test_piecewise_scheduler.py.

            Args:
                epoch: Current training epoch.

            Returns:
                Learning rate value for the given epoch.
            """
            # START TODO ################
            for i, ep in enumerate(self.epochs):
                if epoch <= ep:
                    return self.learning_rates[i]

            return self.learning_rates[-1]
            # END TODO ################

        return lr_lambda


class CosineAnnealingLR(LambdaLR):

    def __init__(self, optimizer: Optimizer, T_max: int):
        """Set learning rate as a cosine decay.

        This class inherits from LambdaLR and implements the lambda
        function that maps the current epoch to the learning rate
        according to epochs.

        Args:
            optimizer: The optimizer to wrap
            T_max:  Maximum number of epochs.
        """
        super().__init__(optimizer, self._get_lambda())
        self.T_max = T_max
        self.initial_lr = self.optimizer.lr

    def _get_lambda(self) -> Callable[[int], float]:
        """Create the function to determine the learning rate given the epoch.

            Returns:
                Lambda LR function.
        """

        def lr_lambda(epoch: int) -> float:
            """Return learning rate given epoch.

            Args:
                epoch: Current training epoch.

            Returns:
                 Learning rate value for given epoch.
            """
            # START TODO ################
            lr_new = 0.5 * (1 + np.cos((epoch * np.pi) / self.T_max)) * self.initial_lr
            return lr_new
            # END TODO ################

        return lr_lambda
