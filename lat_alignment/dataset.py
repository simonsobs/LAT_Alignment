"""
Dataclasses for storing datasets.
"""

from typing import Self
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from numpy.typing import NDArray
import numpy as np

@dataclass
class Dataset:
    """
    Container class for photogrammetry and laser tracker dataset.
    Provides a dict like interface for accessing points by their labels.

    Attributes
    ----------
    data_dict : dict[str, NDArray[np.float64]]
        Dict of data points.
        You should genrally not touch this directly.
    """

    data_dict: dict[str, NDArray[np.float64]]

    def _clear_cache(self):
        self.__dict__.pop("points", None)
        self.__dict__.pop("labels", None)

    def __setattr__(self, name, value):
        if name == "data_dict":
            self._clear_cache()
        return super().__setattr__(name, value)

    def __setitem__(self, key, item):
        self._clear_cache()
        self.data_dict[key] = item

    def __getitem__(self, key):
        return self.data_dict[key]

    def __repr__(self):
        return repr(self.data_dict)

    def __len__(self):
        return len(self.data_dict)

    def __delitem__(self, key):
        self._clear_cache()
        del self.data_dict[key]

    def __contains__(self, item):
        return item in self.data_dict

    def __iter__(self):
        return iter(self.data_dict)

    @cached_property
    def points(self) -> NDArray[np.float64]:
        """
        Get all points in the dataset as an array.
        This is cached.
        """
        return np.array(list(self.data_dict.values()))

    @cached_property
    def labels(self) -> NDArray[np.str_]:
        """
        Get all labels in the dataset as an array.
        This is cached.
        """
        return np.array(list(self.data_dict.keys()))


    def copy(self) -> Self:
        """
        Make a deep copy of the dataset.

        Returns
        -------
        copy : Dataset
            A deep copy of this dataset.
        """
        return deepcopy(self)

@dataclass
class DatasetPhotogrammetry(Dataset):
    """
    Container class for photogrammetry dataset.
    Provides a dict like interface for accessing points by their labels.

    Attributes
    ----------
    data_dict : dict[str, NDArray[np.float64]]
        Dict of photogrammetry points.
        You should genrally not touch this directly.
    """

    def _clear_cache(self):
        self.__dict__.pop("points", None)
        self.__dict__.pop("labels", None)
        self.__dict__.pop("codes", None)
        self.__dict__.pop("code_labels", None)
        self.__dict__.pop("target", None)
        self.__dict__.pop("target_labels", None)

    def __setattr__(self, name, value):
        if name == "data_dict":
            self._clear_cache()
        return super().__setattr__(name, value)

    @cached_property
    def codes(self) -> NDArray[np.float64]:
        """
        Get all coded points in the dataset as an array.
        This is cached.
        """
        msk = np.char.find(self.labels, "CODE") >= 0
        return self.points[msk]

    @cached_property
    def code_labels(self) -> NDArray[np.str_]:
        """
        Get all coded labels in the dataset as an array.
        This is cached.
        """
        msk = np.char.find(self.labels, "CODE") >= 0
        return self.labels[msk]

    @cached_property
    def targets(self) -> NDArray[np.float64]:
        """
        Get all target points in the dataset as an array.
        This is cached.
        """
        msk = np.char.find(self.labels, "TARGET") >= 0
        return self.points[msk]

    @cached_property
    def target_labels(self) -> NDArray[np.str_]:
        """
        Get all target labels in the dataset as an array.
        This is cached.
        """
        msk = np.char.find(self.labels, "TARGET") >= 0
        return self.labels[msk]

    def copy(self) -> Self:
        """
        Make a deep copy of the dataset.

        Returns
        -------
        copy : Dataset
            A deep copy of this dataset.
        """
        return deepcopy(self)
