import numpy as np
import torch
from torch import FloatTensor as ft
from typing import Tuple, List


class ReplayMemory:
    def __init__(self, transition_size: List[int], maxlen: int):
        """
        Vectorized implementation of Replay Memory.

        Example:
        --------
        transition - (state, action, reward, done, new_state)
        transition_size - [4, 1, 1, 1, 4]

        :param transition_size: Length of each of the data in the list.
        :param maxlen: Buffer max length.
        """

        self.maxlen = maxlen
        self.buff = np.empty((maxlen, np.sum(transition_size)))
        self.buff_pointer = 0
        self.buff_split_indices = np.cumsum(transition_size)
        self.passed_size = False

    def push(self, state: np.ndarray, action: object, reward: float, done: bool, next_state: np.ndarray):
        self.buff[self.buff_pointer] = np.hstack((state, action, reward, done, next_state))
        self.buff_pointer += 1
        if self.buff_pointer == self.maxlen:
            self.buff_pointer = 0
            self.passed_size = True

    def get_batch(self, batch_size: int) -> Tuple[ft, torch.Tensor, ft, ft, ft]:
        indices = np.random.randint(0, self.maxlen if self.passed_size else self.buff_pointer, size=batch_size)
        splits = np.split(self.buff[indices], self.buff_split_indices, axis=1)

        return ft(splits[0]), torch.as_tensor(splits[1], dtype=torch.float), ft(splits[2]), ft(splits[3]), ft(splits[4])

    def __len__(self) -> int:
        return self.maxlen if self.passed_size else self.buff_pointer
