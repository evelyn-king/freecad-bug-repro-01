# Copyright (c) Microsoft Corporation. All rights reserved.

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sys
from typing import NamedTuple

from freecad import app as FreeCAD
import Mesh
import Part
import numpy as np


_FREECAD_DEBUG = False


@contextmanager
def silent_stdout():
    sys.stdout.flush()
    stored_py = sys.stdout
    stored_fileno = None
    try:
        stored_fileno = os.dup(sys.stdout.fileno())
    except Exception:
        pass
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        sys.stdout = devnull  # for python stdout.write
        os.dup2(devnull.fileno(), 1)  # for library write to fileno 1
        try:
            yield
        finally:
            sys.stdout = stored_py
            if stored_fileno is not None:
                os.dup2(stored_fileno, 1)


def set_fc_debug(debug: bool):
    global _FREECAD_DEBUG
    _FREECAD_DEBUG = debug


def get_fc_debug() -> bool:
    return _FREECAD_DEBUG


@contextmanager
def fc_debug_context(debug: bool):
    before = get_fc_debug()
    set_fc_debug(debug)
    yield
    set_fc_debug(before)


@contextmanager
def fix_FCDoc():
    with fc_debug_context(True):
        doc = FreeCAD.newDocument("testDoc")
        yield doc
        FreeCAD.closeDocument("testDoc")


class GeometryError(Exception):
    """General geometry error has happened."""


class InvalidShapeError(GeometryError):
    """The Part.Shape or Part.Part2DObject is invalid."""


class ZeroVolumeError(GeometryError):
    """The shape has zero-volume, which should not happen here."""


def is_valid_non_zero_volume(shape: Part.Shape) -> bool:
    """Check if shape is valid and has a non-zero volume."""
    try:
        assert_valid_non_zero_volume(shape)
    except GeometryError:
        return False
    return True


def assert_valid_non_zero_volume(shape: Part.Shape):
    """Check if shape is valid and has a non-zero volume."""
    assert_is_valid(shape)
    assert_non_zero_volume(shape)


def assert_is_valid(part_or_shape: Part.Shape | Part.Feature):
    """Raise a InvalidShapeError if the part or shape is invalid."""
    if isinstance(part_or_shape, Part.Part2DObject):
        # Do not check the Shape of a 2D object, rather just the 2D obj itself
        pass
    elif isinstance(part_or_shape, Part.Feature):
        # Otherwise, if we pass a feature, we take the Shape attribute
        part_or_shape = part_or_shape.Shape
    else:
        assert isinstance(part_or_shape, Part.Shape)

    try:
        is_valid = part_or_shape.isValid()
    except Part.OCCError:
        is_valid = False
    if not is_valid:
        raise InvalidShapeError("Invalid shape.")


def assert_non_zero_volume(shape: Part.Shape):
    """Raise a ZeroVolume error if the shape has a zero-volume."""
    if np.isclose(shape.Volume, 0.0):
        raise ZeroVolumeError("Shape has zero volume.")


class BoundingBox(NamedTuple):
    """Bounding box namedtuple."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @property
    def dx(self) -> float:
        """Length in x."""
        return self.x_max - self.x_min

    @property
    def dy(self) -> float:
        """Width in y."""
        return self.y_max - self.y_min

    @property
    def dz(self) -> float:
        """Height in z."""
        return self.z_max - self.z_min


def makeBB(BB: BoundingBox) -> Part.Box:
    """Make a bounding box given BB tuple.

    Parameters
    ----------
    BB
        Bounding box tuple.

    Returns
    -------
    box
    """
    vec = FreeCAD.Vector
    if type(BB) is tuple:  # Cast to namedtuple
        BB = BoundingBox(*BB)
    doc = FreeCAD.ActiveDocument
    box = doc.addObject("Part::Box")
    centerVector = vec(BB.x_min, BB.y_min, BB.z_min)
    box.Placement = FreeCAD.Placement(
        centerVector, FreeCAD.Rotation(vec(0.0, 0.0, 0.0), 0.0)
    )
    box.Length = BB.dx
    box.Width = BB.dy
    box.Height = BB.dz
    doc.recompute()
    if get_fc_debug():
        assert_valid_non_zero_volume(box.Shape)
    return box



def exportMeshed(obj_list: list[Part.Feature], file_name: str) -> None:
    if not isinstance(obj_list, list):
        raise TypeError("obj_list must be a list of objects.")
    supported_ext = ".stl"
    if file_name.endswith(supported_ext):
        with silent_stdout():
            Mesh.export(obj_list, file_name)
    else:
        raise ValueError(
            file_name
            + " is not a supported extension ("
            + ", ".join(supported_ext)
            + ")"
        )


def test_exportMeshed(cleanup: bool = True):
    with fix_FCDoc() as doc:
        filePath = Path(__file__).parent / "testExport.stl"
        testBB = (-1.0, 1.0, -2.0, 2.0, -3.0, 3.0)
        testShape = makeBB(testBB)
        exportMeshed([testShape], str(filePath))
        Mesh.insert(str(filePath), "testDoc")
        meshImport = doc.getObject("testExport")

        xMin = meshImport.Mesh.BoundBox.XMin
        xMax = meshImport.Mesh.BoundBox.XMax
        yMin = meshImport.Mesh.BoundBox.YMin
        yMax = meshImport.Mesh.BoundBox.YMax
        zMin = meshImport.Mesh.BoundBox.ZMin
        zMax = meshImport.Mesh.BoundBox.ZMax

        meshBB = (xMin, xMax, yMin, yMax, zMin, zMax)
        try:
            assert testBB == meshBB
            if cleanup:
                filePath.unlink()
        except AssertionError:
            print("Test failed")
            print(f"{testBB=}")
            print(f"{meshBB=}")
            raise
        print("Test passed")


if __name__ == "__main__":
    test_exportMeshed()
