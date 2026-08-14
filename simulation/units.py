from __future__ import annotations

import logging
from typing import Any

import awkward as ak
import pint
import pyg4ometry as pg4
from lgdo import LGDO
from lgdo.types import VectorOfVectors
from numpy.typing import ArrayLike

log = logging.getLogger(__name__)

ureg = pint.get_application_registry()
"""The physical units registry."""

# default pretty printing of physical units
ureg.formatter.default_format = "~P"


def pg4_to_pint(obj: pint.Quantity | pg4.gdml.Defines.VectorBase) -> pint.Quantity:
    """Convert pyg4ometry object to pint Quantity."""
    if isinstance(obj, pint.Quantity):
        return obj
    if isinstance(obj, pg4.gdml.Defines.VectorBase):
        return [getattr(obj, field).eval() for field in ("x", "y", "z")] * ureg(obj.unit)
    msg = f"I don't know how to convert object of type {type(obj)} to pint object"
    raise ValueError(msg)


def units_convfact(data: Any | LGDO | ak.Array, target_units: pint.Unit | str) -> float:
    """Calculate numeric conversion factor to reach `target_units`.

    Parameters
    ----------
    data
        starting data structure. If an :class:`LGDO` or :class:`ak.Array`, try to
        determine units by peeking into its attributes. Otherwise, just return 1.
    target_units
        units you wish to convert data to.
    """
    if isinstance(data, LGDO) and "units" in data.attrs:
        return ureg(data.attrs["units"]).to(target_units).magnitude
    if isinstance(data, ak.Array) and "units" in ak.parameters(data):
        return ureg(ak.parameters(data)["units"]).to(target_units).magnitude
    return 1


def attach_units(data: ak.Array | LGDO, unit: str | None) -> ak.Array | LGDO:
    """Convenience function to attach units to `ak.Array` or LGDO.

    Parameters
    ----------
    data
        the array to add units to
    unit
        the unit
    """
    if isinstance(data, ak.Array):
        if unit is not None:
            return ak.with_parameter(data, parameter="units", value=unit)
    elif isinstance(data, LGDO):
        if unit is not None:
            data.attrs["units"] = unit
    else:
        msg = f"to attach units data must be an ak.Array or LGDO not {type(data)}"
        raise TypeError(msg)

    return data


def move_units_to_flattened_data(data: LGDO) -> None:
    """If `data` is a VectorOfVectors move units from attrs to flattened data attrs.

    Parameters
    ----------
    data
        the nested data structure
    """
    if isinstance(data, VectorOfVectors) and ("units" in data.attrs):
        unit = data.attrs.pop("units")
        if isinstance(data.flattened_data, VectorOfVectors):
            data.flattened_data.attrs |= {"units": unit}
            move_units_to_flattened_data(data.flattened_data)
        else:
            data.flattened_data.attrs |= {"units": unit}


def units_conv_ak(
    data: ArrayLike | LGDO | ak.Array | None, target_units: pint.Unit | str
) -> ak.Array | None:
    """Calculate numeric conversion factor to reach `target_units`, and apply to data converted to ak.

    Parameters
    ----------
    data
        starting data structure. If an :class:`LGDO` or :class:`ak.Array`, try to
        determine units by peeking into its attributes. Otherwise, return the data
        converted to a plain :class:`ak.Array` (without units). If ``None``, the
        function will also return ``None``.
    target_units
        units you wish to convert data to.
    """
    if data is None:
        return None

    fact = units_convfact(data, target_units)
    if isinstance(data, LGDO) and fact != 1:
        return ak.without_parameters(data.view_as("ak") * fact)
    if isinstance(data, ak.Array) and fact != 1:
        return ak.without_parameters(data * fact)

    # return ak.Array
    if isinstance(data, LGDO):
        return data.view_as("ak")
    if isinstance(data, ak.Array):
        return data
    return ak.Array(data)


def unwrap_lgdo(data: Any | LGDO | ak.Array, library: str = "ak") -> tuple[Any, pint.Unit | None]:
    """Return a view of the data held by the LGDO and its physical units.

    Parameters
    ----------
    data
        the data container. If not an :class:`LGDO` or :class:`ak.Array`, it will be
        returned as is with ``None`` units.
    library
        forwarded to :meth:`LGDO.view_as`.

    Returns
    -------
    A tuple of the un-lgdo'd data and the data units.
    """
    ret_data = data
    ret_units = None
    if isinstance(data, LGDO):
        ret_data = data.view_as(library)
        if "units" in data.attrs:
            ret_units = ureg(data.attrs["units"]).u

    if isinstance(data, ak.Array):
        if library != "ak":
            msg = "cannot unwrap an awkward array as a non-awkward type"
            raise ValueError(msg)

        if "units" in ak.parameters(data):
            ret_units = ureg(ak.parameters(data)["units"]).u
            ret_data = ak.without_parameters(data)

    return ret_data, ret_units


def get_unit_str(data: ak.Array | LGDO) -> str | None:
    """Get the units as a string for an awkward array with attached units.

    Parameters
    ----------
    data
        the array with attached units

    Returns
    -------
    a string of the units.
    """
    if isinstance(data, ak.Array):
        attrs = ak.parameters(data)
    elif isinstance(data, LGDO):
        attrs = data.attrs
    else:
        msg = f"Cannot extract units: {data} is not an LGDO or ak.Array"
        raise ValueError(msg)

    if "units" in attrs:
        return attrs["units"]
    return None


def unit_to_lh5_attr(unit: pint.Unit) -> str:
    """Convert Pint unit to a string that can be used as attrs["units"] in an LGDO."""
    # TODO: we should check if this can be always parsed by Unitful.jl
    if isinstance(unit, pint.Unit):
        return f"{unit:~C}"
    return unit
