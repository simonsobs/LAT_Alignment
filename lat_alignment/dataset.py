"""
Dataclasses for storing datasets.
"""

from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from typing import Self

import numpy as np
from numpy.typing import NDArray


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
        self.__dict__.pop("errs", None)
        self.__dict__.pop("labels", None)
        self.__dict__.pop("targets", None)
        self.__dict__.pop("target_errs", None)
        self.__dict__.pop("target_labels", None)

    def __setattr__(self, name, value):
        if name == "data_dict":
            self._clear_cache()
        return super().__setattr__(name, value)

    def __setitem__(self, key, item):
        self._clear_cache()
        if key[-4:] == "_err":
            self.data_dict[key[:-4]][3:] = item
        self.data_dict[key][:3] = item
        self._clear_cache()

    def __getitem__(self, key):
        if key[-4:] == "_err":
            return self.data_dict[key[:-4]][3:]
        return self.data_dict[key][:3]

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
        return np.array(list(self.data_dict.values()))[:, :3]

    @cached_property
    def errs(self) -> NDArray[np.float64]:
        """
        Get all points in the dataset as an array.
        This is cached.
        """
        return np.array(list(self.data_dict.values()))[:, 3:]

    @cached_property
    def labels(self) -> NDArray[np.str_]:
        """
        Get all labels in the dataset as an array.
        This is cached.
        """
        return np.array(list(self.data_dict.keys()))

    @cached_property
    def targets(self) -> NDArray[np.float64]:
        """
        Get all target points in the dataset as an array.
        This is cached.
        """
        msk = np.char.find(self.labels, "TARGET") >= 0
        return self.points[msk]

    @cached_property
    def target_errs(self) -> NDArray[np.float64]:
        """
        Get all target errors in the dataset as an array.
        This is cached.
        """
        msk = np.char.find(self.labels, "TARGET") >= 0
        return self.errs[msk]

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


@dataclass
class DatasetReference(Dataset):
    """
    Container class for dataset that only contains reference points and their errors.
    Provides a dict like interface for accessing points by their labels.
    Note that this class doesn't cover edge cases from misuse, use at your own risk.

    Attributes
    ----------
    data_dict : dict[str, NDArray[np.float64]]
        Dict of data points.
        You should genrally not touch this directly.
    """

    def _clear_cache(self):
        self.__dict__.pop("points", None)
        self.__dict__.pop("errs", None)
        self.__dict__.pop("labels", None)
        self.__dict__.pop("elem_names", None)
        self.__dict__.pop("elem_labels", None)
        self.__dict__.pop("ref_labels", None)
        self.__dict__.pop("err_labels", None)
        self.__dict__.pop("elements", None)
        self.__dict__.pop("reference", None)
        self.__dict__.pop("errors", None)

    def __setattr__(self, name, value):
        if name == "data_dict":
            self._clear_cache()
        return super().__setattr__(name, value)

    def __setitem__(self, key, item):
        if key in self.elem_names:
            if item.shape != self.elements[key].shape:
                raise ValueError(
                    "Can't set {key} with shape {item.shape} when original shape is {self.elements[key]}"
                )
            for i, l in enumerate(self.elem_labels[key]):
                self.data_dict[l] = item[i]
        elif key[:-4] in self.elem_names and "_ref" in key:
            if item.shape != self.reference[key[:-4]].shape:
                raise ValueError(
                    "Can't set {key} with shape {item.shape} when original shape is {self.reference[key]}"
                )
            for i, l in enumerate(self.ref_labels[key[:-4]]):
                self.data_dict[l] = item[i]
        elif key[:-4] in self.elem_names and "_err" in key:
            if item.shape != self.errors[key[:-4]].shape:
                raise ValueError(
                    "Can't set {key} with shape {item.shape} when original shape is {self.error[key]}"
                )
            for i, l in enumerate(self.err_labels[key[:-4]]):
                self.data_dict[l] = item[i]
        else:
            self.data_dict[key] = item
        self._clear_cache()

    def __getitem__(self, key):
        return self.data_dict[key]

    @cached_property
    def elem_names(self) -> list[str]:
        return np.unique([k.split("_")[0] for k in self.labels]).tolist()

    @cached_property
    def elem_labels(self) -> dict[str, list[str]]:
        return {
            e: [
                k
                for k in self.labels
                if (e in k and k[-4:] != "_ref" and k[-4:] != "_err")
            ]
            for e in self.elem_names
        }

    @cached_property
    def elements(self) -> dict[str, NDArray[np.float64]]:
        return {
            e: np.array([self.data_dict[k] for k in self.elem_labels[e]])
            for e in self.elem_names
        }

    @cached_property
    def ref_labels(self) -> dict[str, list[str]]:
        return {
            e: [k for k in self.labels if (e in k and k[-4:] == "_ref")]
            for e in self.elem_names
        }

    @cached_property
    def reference(self) -> dict[str, NDArray[np.float64]]:
        return {
            e: np.array([self.data_dict[k] for k in self.ref_labels[e]])
            for e in self.elem_names
        }

    @cached_property
    def err_labels(self) -> dict[str, list[str]]:
        return {
            e: [k for k in self.labels if (e in k and k[-4:] == "_err")]
            for e in self.elem_names
        }

    @cached_property
    def errors(self) -> dict[str, NDArray[np.float64]]:
        return {
            e: np.array([self.data_dict[k] for k in self.err_labels[e]])
            for e in self.elem_names
        }


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
        self.__dict__.pop("errs", None)
        self.__dict__.pop("labels", None)
        self.__dict__.pop("codes", None)
        self.__dict__.pop("code_errs", None)
        self.__dict__.pop("code_labels", None)
        self.__dict__.pop("targets", None)
        self.__dict__.pop("target_errs", None)
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
    def code_errs(self) -> NDArray[np.float64]:
        """
        Get all coded point errors in the dataset as an array.
        This is cached.
        """
        msk = np.char.find(self.labels, "CODE") >= 0
        return self.errs[msk]

    @cached_property
    def code_labels(self) -> NDArray[np.str_]:
        """
        Get all coded labels in the dataset as an array.
        This is cached.
        """
        msk = np.char.find(self.labels, "CODE") >= 0
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
