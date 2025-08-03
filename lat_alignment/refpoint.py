"""
Functions and dataclasses to handle reference points.
"""

import logging
import operator
from dataclasses import dataclass
from functools import cached_property, reduce
from typing import Self

import numpy as np
from numpy.typing import NDArray


def _pad_missing(arr1, arr2):
    master = arr1
    to_pad = arr2
    if len(arr2) > len(arr1):
        master = arr2
        to_pad = arr1
    pad_msk = np.zeros_like(master, bool)
    if len(master) == len(to_pad):
        return to_pad, pad_msk
    if not np.isclose(to_pad[0], master[0]):
        pstart = np.where(np.isclose(master, to_pad[0]))[0][0]
        to_pad = np.hstack((master[:pstart], to_pad))
        pad_msk[:pstart] = True
    dmaster = np.diff(master)
    dpad = np.diff(to_pad)
    while not np.allclose(dmaster[: len(dpad)], dpad):
        didx = np.where(~np.isclose(dmaster[: len(dpad)], dpad))[0][0]
        to_insert = master[didx + 1 : didx + 2]
        to_pad = np.hstack((to_pad[: didx + 1], to_insert, to_pad[didx + 1 :]))
        pad_msk[didx + 1 : didx + 2] = True
        dpad = np.diff(to_pad)
        if len(to_pad) == len(master):
            break
    if not np.isclose(to_pad[-1], master[-1]):
        pend = np.where(np.isclose(master, to_pad[-1]))[0][-1] + 1
        to_pad = np.hstack((to_pad, master[pend:]))
        pad_msk[pend:] = True
    return to_pad, pad_msk


@dataclass
class RefTOD:
    """
    Dataclass for storing the position of a point as a function of time.

    Attributes
    ----------
    name : str
        The name of the point.
    data : NDArray[np.float64]
        The position of the point.
        Should be a `(npoint, ndim)` array.
    angle : NDArray[np.float64]
        The angle of the relevent element for each data point.
        For example the corotator angle or elevation angle.
        Should have shape `(npoint,)`.
    """

    name: str
    data: NDArray[np.float64]
    angle: NDArray[np.float64]

    def __setattr__(self, name, val):
        self.__dict__[name] = val
        if name in ["data", "angle"]:
            self.__dict__.pop("npoints", None)
            self.__dict__.pop("meas_number", None)
        if name == "angle":
            self.__dict__.pop("direction", None)

    @cached_property
    def npoints(self) -> int:
        """
        The number of points in the TOD.
        Will throw and error if `data` and `angle` disagree on this.
        """
        if len(self.data) != len(self.angle):
            raise ValueError("Data and angle don't have same length!")
        return len(self.data)

    @cached_property
    def direction(self) -> NDArray[np.float64]:
        """
        The derivative of `self.angle`.
        """
        direction = np.diff(self.angle)
        direction = np.hstack((direction, [direction[-1]]))
        return direction

    @cached_property
    def meas_number(self) -> NDArray[np.integer]:
        """
        Array that indexes the TOD data (ie: sample number).
        """
        return np.arange(self.npoints)


@dataclass
class RefElem:
    """
    Dataclass for storing a collection of `RefTOD`s that belong to the same element (ie: primary, secondary, etc).

    Attributes
    ----------
    name : str
        The name of the element.
    tods : list[RefTOD]
        The `RefTOD`s that make up this element.
        All these TODs must agree on the number of measurements
        as well as the angle at each measurements.
    """

    name: str
    tods: list[RefTOD]

    def __setattr__(self, name, val):
        self.__dict__[name] = val
        if name == "tods":
            self.__dict__.pop("meas_number", None)
            self.__dict__.pop("npoints", None)
            self.__dict__.pop("ntods", None)
            self.__dict__.pop("direction", None)
            self.__dict__.pop("angle", None)
            self.__dict__.pop("data", None)
            self.__dict__.pop("tod_names", None)
            self._check()

    @cached_property
    def data(self) -> NDArray[np.float64]:
        """
        The data for all the `RefTOD`s in this element.
        Will have shape `(npoint, ntod, ndim)`.
        """
        data = np.swapaxes(
            np.atleast_3d(np.array([e.data for e in self.tods])),
            0,
            1,
        )
        return data

    @cached_property
    def angle(self) -> NDArray[np.float64]:
        """
        The angle for all the `RefTOD`s in this element.
        Will have shape `(npoint,)`.
        """
        return self.tods[0].angle

    @cached_property
    def direction(self) -> NDArray[np.float64]:
        """
        The direction for all the `RefTOD`s in this element.
        Will have shape `(npoint,)`.
        """
        return self.tods[0].direction

    @cached_property
    def npoints(self) -> int:
        """
        The number of points in each `RefTOD`.
        """
        return self.tods[0].npoints

    @cached_property
    def ntods(self) -> int:
        """
        The number of `RefTOD`s in this element.
        """
        return len(self.tods)

    @cached_property
    def meas_number(self) -> NDArray[np.integer]:
        """
        Array that indexes the TOD data (ie: sample number).
        """
        return self.tods[0].meas_number

    @cached_property
    def tod_names(self) -> NDArray[np.str_]:
        """
        The names of the `RefTOD`s in this element.
        """
        return np.array([t.name for t in self.tods])

    def _check(self):
        if len(self.tods) == 0:
            raise ValueError("Empty element!")
        if len(np.unique(self.tod_names)) != len(self.tods):
            raise ValueError("TOD names not uniqe!")
        all_npoints = np.array([t.npoints for t in self.tods])
        if np.any(all_npoints != self.npoints):
            raise ValueError("Not all TODs have the same length!")
        all_ang = np.array([e.angle for e in self.tods])
        if not (np.isclose(all_ang, self.angle) | np.isnan(all_ang)).all():
            raise ValueError("Angles don't agree!")
        all_dir = np.array([e.direction for e in self.tods])
        if not (np.isclose(all_dir, self.direction) | np.isnan(all_dir)).all():
            raise ValueError("Directions don't agree!")

    def reorder(self, names: NDArray[np.str_], pad: bool = False) -> Self:
        """
        Reorder the `RefTOD`s in this element.

        Parameters
        ----------
        names : NDArray[np.str_]
            The names in the requested order.
        pad : bool, default: False
            If True then if there are names not found in this
            element they will be added with `np.nan` for all data.

        Returns
        -------
        reordered : Self
            The reordered `RefElem`.
            The object is also modified in place.

        Raises
        ------
        ValueError
            If the element has no TODs.
            If `names` is not unique.
            If we aren't padding and `names` isn't a subset of `self.tod_names`.
        """
        if len(self.tods) == 0:
            raise ValueError("Can't reorder empty element!")
        names = np.array(names)
        if len(np.unique(names)) != len(names):
            raise ValueError("Input names not unique")
        inself = np.isin(names, self.tod_names)
        tods = self.tods
        if np.sum(inself) != len(names):
            if not pad:
                names = names[inself]
            else:
                null_dat = np.zeros_like(tods[0].data, np.float64) + np.nan
                tods += [
                    RefTOD(n, null_dat.copy(), self.angle.copy())
                    for n in names[~inself]
                ]
        tods = [t for t in tods if t.name in names]
        if len(tods) != len(names):
            raise ValueError("Can't find enough TODs with the input names!")
        tod_names = np.array([t.name for t in tods])
        mapping = np.argsort(np.argsort(names))
        nsrt = np.argsort(tod_names)
        tods_srt = [tods[i] for i in nsrt]
        tods_srt = [tods_srt[i] for i in mapping]

        self.tods = tods_srt
        return self


@dataclass
class RefCollection:
    """
    Dataclass for storing a collection of `RefElem`s with measurements taken together.

    Attributes
    ----------
    elems : list[RefElem]
        The `RefElem`s in the collection.
    """

    elems: list[RefElem]

    def __setattr__(self, name, val):
        self.__dict__[name] = val
        if name == "elems":
            self.__dict__.pop("meas_number", None)
            self.__dict__.pop("npoints", None)
            self.__dict__.pop("nelems", None)
            self.__dict__.pop("direction", None)
            self.__dict__.pop("angle", None)
            self.__dict__.pop("elem_names", None)
            self.__dict__.pop("_elem_dict", None)
            self._check()

    @cached_property
    def elem_names(self) -> list[str]:
        """
        The names of the `RefElem`s in the collection.
        """
        return [e.name for e in self.elems]

    @cached_property
    def _elem_dict(self):
        return {n: e for n, e in zip(self.elem_names, self.elems)}

    def __getitem__(self, index):
        return self._elem_dict[index]

    def keys(self):
        return self._elem_dict.keys()

    def values(self):
        return self._elem_dict.values()

    def items(self):
        return self._elem_dict.items()

    @cached_property
    def angle(self) -> NDArray[np.float64]:
        """
        The angle for all the `RefElems`s in this element.
        Will have shape `(npoint,)`.
        """
        return self.elems[0].angle

    @cached_property
    def direction(self) -> NDArray[np.float64]:
        """
        The direction for all the `RefElems`s in this element.
        Will have shape `(npoint,)`.
        """
        return self.elems[0].direction

    @cached_property
    def npoints(self) -> int:
        """
        The number of points in each `RefTOD`.
        """
        return self.elems[0].npoints

    @cached_property
    def nelems(self) -> int:
        """
        The number of elems in each `RefElem`.
        """
        return len(self.elems)

    @cached_property
    def meas_number(self) -> NDArray[np.integer]:
        """
        Array that indexes the TOD data (ie: sample number).
        """
        return self.elems[0].meas_number

    def _check(self):
        for elem in self.values():
            elem._check()
            if not (np.isclose(elem.angle, self.angle) | np.isnan(elem.angle)).all():
                raise ValueError("Angles don't agree!")
            if not (
                np.isclose(elem.direction, self.direction) | np.isnan(elem.direction)
            ).all():
                raise ValueError("Directions don't agree!")

    @classmethod
    def construct(
        cls, data: dict[str, list[RefTOD]], logger: logging.Logger, pad: bool = False
    ) -> Self:
        """
        Construct a `RefCollection` from `RefTOD`s.

        Parameters
        ----------
        data : dict[str, list[RefTOD]]
            The data to construct from.
            Each item in the `dict` should be a list of `RefTOD`s that are from the same element,
            with the key being the element name.
        logger : logging.Logger
            The logger object to use.
        pad : bool, default: False
            If True then attempt to pad the `RefTOD` so they agree on the angle and measurements number.

        Returns
        -------
        collection : RefCollection
            The constructed `RefCollection`.
        """
        npoints = np.hstack(
            [[point.npoints for point in data[elem]] for elem in data.keys()]
        )
        if not np.all(npoints == npoints[0]):
            if not pad:
                raise ValueError("Not all points have the same number of measurements!")
            logger.warning("\tPadding data with nans")
            master_angle = [
                [point.angle for point in data[elem]] for elem in data.keys()
            ]
            master_angle = reduce(operator.iconcat, master_angle, [])
            nangs = np.array([len(ang) for ang in master_angle])
            master_angle = master_angle[np.argmax(nangs)]
            for elem in data.keys():
                for i, point in enumerate(data[elem]):
                    ang_pad, pad_msk = _pad_missing(master_angle, point.angle)
                    dat_pad = np.zeros((len(ang_pad),) + point.data.shape[1:]) + np.nan
                    dat_pad[~pad_msk] = point.data
                    data[elem][i] = RefTOD(point.name, dat_pad, ang_pad)
        elems = []
        for elem in data.keys():
            logger.info("\tConstructing TOD for %s", elem)
            try:
                relem = RefElem(elem, data[elem])
                relem._check()
            except ValueError as e:
                logger.error(f"\t\tFailed with error: {e}. Skipping...")
                continue
            if relem.data.size == 0:
                logger.info("\t\tNo data found! Not making TOD")
                continue
            elems += [relem]

        return cls(elems)
