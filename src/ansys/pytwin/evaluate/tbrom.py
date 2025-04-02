# Copyright (C) 2022 - 2025 ANSYS, Inc. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import os
from pathlib import Path
import struct
from typing import TYPE_CHECKING, Union

import numpy as np
from pytwin import _HAS_TQDM
from pytwin.decorators import needs_graphics

if TYPE_CHECKING:  # pragma: no cover
    import pyvista as pv


def read_binary(filepath):
    """
    Read a binary snapshot file from the disk.

    Parameters
    ----------
    filepath : str
        Path of the binary file to be read.

    Returns
    -------
    np.ndarray
        Return a 1D flatenned Numpy array of snapshot data read.

    Examples
    --------
    >>> from pytwin import read_binary
    >>> snapshot_data = read_binary('snapshot.bin')
    """
    return np.fromfile(filepath, dtype=np.double, offset=8).reshape(
        -1,
    )


def write_binary(filepath: str, vec: np.ndarray):
    """
    Write a binary snapshot file on the disk.

    Parameters
    ----------
    filepath : str
        Path of the binary file to be written.
    vec : np.ndarray
        N-dimensional Numpy array of snapshot data to be written in binary file.

    Returns
    -------
    bool
        Return True if the binary file is successfully written.

    Examples
    --------
    >>> from pytwin import write_binary
    >>> scalar_field = np.ndarray([1.0, 2.0, 3.0, 5.0])
    >>> write_binary('snapshot_scalar.bin', scalar_field)
    >>> vector_field = np.ndarray([[1.0, 1.0, 0.0], [1.0, 2.0, 3.0], [5.0, 3.0, 3.0], [5.0, 5.0, 6.0]])
    >>> write_binary('snapshot_vector.bin', vector_field)
    """
    vec = vec.reshape(
        -1,
    )
    if os.path.exists(filepath):
        os.remove(filepath)
    with open(filepath, "xb") as f:
        f.write(struct.pack("Q", len(vec)))
        vec.tofile(f)
    return True


def read_snapshot_size(filepath):
    """
    Return the number of data stored in a snapshot binary file.

    Parameters
    ----------
    filepath : str
        Path of the binary file to be written.

    Returns
    -------
    int
        Number of data stored in the binary file.

    Examples
    --------
    >>> from pytwin import read_snapshot_size
    >>> number_data = read_snapshot_size('snapshot.bin')
    """
    with open(filepath, "rb") as f:
        nbdof = struct.unpack("Q", f.read(8))[0]
    return nbdof


def snapshot_to_array(snapshot_file, geometry_file):
    """
    Create an array containing the x, y, z coordinates and data from geometry
    and field snapshot files.

    Parameters
    ----------
    snapshot_file : str
        Path of the binary field data file to be read.
    geometry_file : str
        Path of the binary points data file to be read.

    Returns
    -------
    np.ndarray
        Return a 2D Numpy array of x,y,z coordinates and snapshot data read.
        Array has shape (m,n), where m is the number of points in the geometry
        file and n is the dimension of the snapshot field + 3.

    Raises
    ------
    ValueError
        if snapshot lengths are incompatible.

    Examples
    --------
    >>> from pytwin import snapshot_to_array
    >>> snapshot_data = snapshot_to_array('snapshot.bin', 'points.bin')
    """
    n_g = read_snapshot_size(geometry_file)
    if n_g % 3 > 0:
        raise ValueError("Geometry snapshot length must be divisible by 3.")
    n_points = n_g // 3
    n_s = read_snapshot_size(snapshot_file)
    if n_s % n_points > 0:
        raise ValueError(f"Field snapshot length {n_s} must be divisible by the number of points {n_points}.")

    geometry_data = read_binary(geometry_file).reshape(-1, 3)
    snapshot_data = read_binary(snapshot_file).reshape(geometry_data.shape[0], -1)
    return np.concatenate((geometry_data, snapshot_data), axis=1)


def _read_basis(filepath):
    with open(filepath, "rb") as f:
        var = struct.unpack("cccccccccccccccc", f.read(16))[0]
        nb_val = struct.unpack("Q", f.read(8))[0]
        nb_mc = struct.unpack("Q", f.read(8))[0]
        return np.fromfile(f, dtype=np.double, offset=0).reshape(-1, nb_val)


def _read_settings(filepath):
    with open(filepath) as f:
        data = json.load(f)

    namedselection = {}
    dimensionality = None
    name = None
    outputname = None
    unit = None

    if "namedSelections" in data:
        namedselection = data["namedSelections"]
    if "dimensionality" in data:
        dimensionality = data["dimensionality"]
        if len(dimensionality) > 1:  # tensor field
            if data["symmetricalDim"]:  # symmetric tensor -> 6 components
                dimensionality = [6]
            else:  # non symmetric tensor -> 9 components
                dimensionality = [9]

    if "name" in data:
        name = data["name"]
        outputname = name.replace(" ", "_")
    if "unit" in data:
        unit = data["unit"]

    tbromns = dict()

    # Create list of name selections indexes
    for name, idsList in namedselection.items():
        idsListNp = np.array(idsList)
        ind = np.where(idsListNp == -1)
        i = 0
        for elem in np.nditer(ind):
            subarray = np.arange(idsListNp[elem - 1 - i] + 1, idsListNp[elem + 1 - i])
            idsListNp = np.delete(idsListNp, elem - i)
            idsListNp = np.concatenate((idsListNp, subarray))
            i = i + 1
        tbromns.update({name: idsListNp})

    return [tbromns, dimensionality, outputname, unit]


def _read_properties(filepath):
    with open(filepath) as f:
        data = json.load(f)

    fields = {}
    if "fields" in data:
        fields = data["fields"]
    out_field = fields["outField"]

    nb_points = out_field["nbDof"]
    nb_modes = out_field["nbModes"]

    return [nb_points, nb_modes]


class TbRom:
    """
    Instantiates a TBROM that is part of a TWIN file created by Ansys Twin Builder.

    After a twin model is initialized and its TBROM is instantiated, additional functionalities
    are available to generate snapshots (in memory or on disk) and project input field data
    (if the TBROM is parameterized with an input field).

    Parameters
    ----------
    tbrom_name : str
        Name of the TBROM included in the TWIN file (it is the name of the TBROM component inserted in the Twin Builder
        subsheet used to assemble the TWIN model).
    tbrom_path : str
        File path to the folder with TBROM files (it is the temporary directory used by the Twin Runtime SDK for the
        evaluation of this TBROM).
    """

    IN_F_KEY = "binaryInputField"
    OUT_F_KEY = "binaryOutputField"
    TBROM_BASIS = "basis.svd"
    TBROM_SET = "settings.json"
    TBROM_PROP = "properties.json"
    TBROM_POINTS = "points.bin"

    def __init__(self, tbrom_name: str, tbrom_path: str):
        self._tbrom_path = tbrom_path
        self._name = tbrom_name
        self._infmcs = None
        self._outmcs = None
        self._infbasis = None
        self._pointsdata = None
        self._meshdata = None
        self._outbasis = None
        self._outmeshbasis = None
        self._hasinfmcs = None
        self._hasoutmcs = False

        files = os.listdir(tbrom_path)
        infdata = dict()

        for file in files:
            if TbRom.IN_F_KEY in file:
                folder = file.split("_")
                fname = folder[1]
                inpath = os.path.join(tbrom_path, file, TbRom.TBROM_BASIS)
                inbasis = _read_basis(inpath)
                infdata.update({fname: inbasis})
        self._infbasis = infdata

        settingspath = os.path.join(tbrom_path, TbRom.OUT_F_KEY, TbRom.TBROM_SET)
        [nsidslist, dimensionality, outputname, unit] = _read_settings(settingspath)
        self._nsidslist = nsidslist
        self._outdim = int(dimensionality[0])
        self._outname = outputname
        self._outunit = unit
        self._outputfilespath = None

        propertiespath = os.path.join(tbrom_path, TbRom.TBROM_PROP)
        [nbpoints, nbmodes] = _read_properties(propertiespath)
        # bug 1168769 (fixed in 2025R2)
        pointpath = os.path.join(tbrom_path, TbRom.OUT_F_KEY, TbRom.TBROM_POINTS)
        if os.path.exists(pointpath):
            self._nbpoints = read_snapshot_size(pointpath) // 3
        else:
            self._nbpoints = int(nbpoints / self._outdim)
        self._nbmodes = nbmodes

        self._has_point_file = self._read_points(pointpath)

        outpath = os.path.join(tbrom_path, TbRom.OUT_F_KEY, TbRom.TBROM_BASIS)
        self._init_pointsdata(outpath)

    def _generate_points(self, on_disk: bool, output_file_path: str, named_selection: str = None):
        """
        Generate a point file for the full field or a specific part.

        The point file can be saved in memory or written to disk. When it is written to disk,
        the path specified for the ``output_file_path`` parameter is used.

        Parameters
        ----------
        on_disk : bool
            Whether the point file is saved on disk (True) or returned in memory (False).
        output_file_path: str
            Path for where the point file is written when saved to disk.
        named_selection: str (optional)
            Named selection on which the point file has to be generated. The default is ``None``, in which case the
            entire domain is considered.
        """
        vec = self._data_extract(named_selection, self._pointsdata.points)
        if on_disk:
            write_binary(output_file_path, vec)
            return output_file_path
        else:
            return vec

    def _generate_snapshot(self, on_disk: bool, output_file_path: str, named_selection: str = None):
        """
        Generate a field snapshot based on current states of the TBROM for the
        full field or a specific part.

        The field snapshot can be saved in memory or written to disk. When it is
        written to disk, the path specified for the ``output_file_name`` parameter
        is used.

        Parameters
        ----------
        on_disk: bool
            Whether the snapshot file is saved on disk (True) or returned in memory (False)
        output_file_path: str
            Path where the snapshot file is written if on_disk is True
        named_selection: str (optional)
            Named selection on which the snasphot has to be generated. The default is ``None``, in which case the
            entire domain is considered.
        """
        vec = self._data_extract(named_selection, self._pointsdata[self.field_output_name])
        if on_disk:
            write_binary(output_file_path, vec)
            return output_file_path
        else:
            return vec

    def _reduce_field_input(self, name: str, snapshot: Union[str, Path, np.ndarray]):
        """
        Project a snapshot associated to the input field name ``fieldname``

        Parameters
        ----------
        name: str
            Name of the input field to project the snapshot. The name of the field must be specified in case the TBROM
            is parameterized with multiple input fields.
        snapshot: str | Path | np.ndarray
            Path of the input field snapshot file, or numpy array of snapshot data
        """
        mc = []
        if isinstance(snapshot, np.ndarray):
            vecnp = snapshot
        else:
            vecnp = read_binary(snapshot)

        if name is None or self.field_input_count == 1:
            basis = list(self._infbasis.values())[0]
        else:
            basis = self._infbasis[name]
        nb_mc = len(basis)
        for i in range(nb_mc):
            mnp = np.array(basis[i])
            mci = mnp.dot(vecnp)
            mc.append(mci)
        if name is None or self.field_input_count == 1:
            index = 0
            for item, key in self._infmcs[self.field_input_names[0]].items():
                self._infmcs[self.field_input_names[0]][item] = mc[index]
                index = index + 1
        else:
            index = 0
            for item, key in self._infmcs[name].items():
                self._infmcs[name][item] = mc[index]
                index = index + 1

    def _project_on_mesh(
        self,
        target_mesh: "pv.DataSet",
        interpolate: bool,
        named_selection: str = None,
        nodal_values: bool = False,
        sharpness: float = 5.0,
        radius: float = 0.0001,
        strategy: str = "closest_point",
        null_value: float = 0.0,
        n_points: int = None,
        all_points: bool = False,
    ):
        """
        Project the field ROM SVD basis onto a targeted mesh.

        Parameters
        ----------
        target_mesh: pyvista.DataSet
            PyVista DataSet object of the targeted mesh.
        interpolate: bool
            Flag to indicate whether the point cloud data are interpolated (True) or not (False) on the targeted mesh.
        named_selection: str (optional)
            Named selection on which the mesh projection has to be performed. The default is ``None``, in which case the
            entire domain is considered.
        nodal_values: bool (optional)
            Control whether the interpolated results are returned as nodal values, or cell values (default)
        sharpness: float, default: 5.0
            Set the sharpness (i.e., falloff) of the Gaussian interpolation kernel.
        radius: float, default: 0.0001
            Specify the radius within which the basis points must lie.
        strategy: str, default: "closest_point"
            Specify a strategy to use when encountering a "null" point during the interpolation process. Valid values
            are ``'null_value'``, ``'mask_points'`` and ``'closest_point'``. If the ``'mask_points'`` strategy is used
            then only non-null points are retained in the projected SVD basis.
        null_value: float, default: 0.0
            Specify the null point value.
        n_points: int, optional
            If given, specifies the number of the closest points used to form the interpolation basis.
        all_points: bool, default: False
            When ``strategy='mask_points'``, when this value is ``True`` only cells where all points are valid are kept.
            When ``False`` cells are kept if any of their points are valid and invalid points are given the
            ``null_value``.

        Raises
        ------
        ValueError
            If masking interpolation strategy finds no valid points.

        See Also
        --------
        pyvista.DataSetFilters.interpolate :
            Detailed description of ``sharpness``, ``radius``, ``strategy``, ``null_value`` and ``n_points`` parameters.
        """
        nbmc = self.nb_modes
        if not interpolate:  # target mesh is same as the one used to generate the ROM -> no interpolation required
            outmeshbasis = self._outbasis
            if named_selection is not None:
                pointsids = self._named_selection_indexes(named_selection)
                listids = np.sort(pointsids)
                outmeshbasis = outmeshbasis[:, listids]
            self._outmeshbasis = outmeshbasis
        else:  # interpolation required, e.g. because target mesh is different
            progress_bar = False
            if _HAS_TQDM:
                progress_bar = True
            pointsdata = self._pointsdata.copy()
            for i in range(0, nbmc):
                pointsdata[str(i)] = self._outbasis[i]
            if named_selection is not None:
                pointsids = self._named_selection_indexes(named_selection)
                listids = np.sort(pointsids)
                pointsdata = pointsdata.extract_points(listids)
            interpolated_points = target_mesh.interpolate(
                pointsdata,
                sharpness=sharpness,
                radius=radius,
                strategy=strategy,
                null_value=null_value,
                n_points=n_points,
                progress_bar=progress_bar,
            )
            if strategy == "mask_points":
                interpolated_points = interpolated_points.threshold(
                    value=0.5, scalars="vtkValidPointMask", preference="point", all_scalars=all_points
                )
                if interpolated_points.n_cells == 0:
                    raise ValueError(
                        "[TbRomInterpolation]No valid points found. Check mesh size and interpolation settings"
                    )
            if not nodal_values:
                interpolated_mesh = interpolated_points.point_data_to_cell_data()
                self._outmeshbasis = np.array([interpolated_mesh.cell_data[str(i)] for i in range(0, nbmc)])
            else:
                interpolated_mesh = interpolated_points
                self._outmeshbasis = np.array([interpolated_mesh.point_data[str(i)] for i in range(0, nbmc)])

        if interpolate and strategy == "mask_points":
            mesh_data = interpolated_mesh.copy()
        else:
            mesh_data = target_mesh.copy()
        # Clear the mesh data to strip out any existing field data
        mesh_data.clear_data()

        # initialize output field data
        nb_data = self._outmeshbasis.shape[1]
        mesh_data[self.field_output_name] = np.zeros((nb_data, self.field_output_dim))

        self._meshdata = mesh_data

    def _update_output_field(self):
        """
        Compute the output field results with current mode coefficients.
        """
        mc = np.asarray(list(self._outmcs.values()))

        self._pointsdata[self.field_output_name] = np.tensordot(mc, self._outbasis, axes=1)
        self._pointsdata.set_active_scalars(self.field_output_name)

        if self._meshdata is not None:
            self._meshdata[self.field_output_name] = np.tensordot(mc, self._outmeshbasis, axes=1)
            self._meshdata.set_active_scalars(self.field_output_name)

    def _named_selection_indexes(self, nsname: str):
        return self._nsidslist[nsname]

    def _input_field_size(self, fieldname: str):
        return len(self._infbasis[fieldname][0])

    def _data_extract(self, named_selection: str, data: np.ndarray):
        if named_selection is not None:
            pointsids = self._named_selection_indexes(named_selection)
            listids = np.sort(pointsids)
            return data[listids]
        else:
            return data

    @needs_graphics
    def _read_points(self, filepath):
        import pyvista as pv

        if os.path.exists(filepath):
            points = read_binary(filepath)
            has_point_file = True
        else:
            points = np.zeros(3 * self.nb_points)
            has_point_file = False
        self._pointsdata = pv.PolyData(points.reshape(-1, 3))
        return has_point_file

    def _init_pointsdata(self, filepath):
        self._outbasis = _read_basis(filepath).reshape(self.nb_modes, self.nb_points, self.field_output_dim)
        # initialize output field data
        if self._hasoutmcs:
            self._pointsdata[self.field_output_name] = np.zeros((self.nb_points, self.field_output_dim))

    @property
    def has_point_file(self):
        return self._has_point_file

    @property
    def field_input_count(self):
        """Return the number of field input(s) that can be used for this TBROM."""
        return len(self._infbasis)

    @property
    def field_input_names(self):
        """Return a list of the input field names that can be used for this TBROM."""
        if self._infbasis is not None:
            return list(self._infbasis.keys())
        else:
            return []

    @property
    def field_output_dim(self):
        """Return the dimension of the field output."""
        return self._outdim

    @property
    def field_output_name(self):
        """Return the field output name."""
        return self._outname

    @property
    def field_output_unit(self):
        """Return the field output unit."""
        return self._outunit

    @property
    def named_selections(self):
        """Return the list of named selections available for this TBROM."""
        return list(self._nsidslist.keys())

    @property
    def name(self):
        return self._name

    @property
    def field_on_mesh(self):
        """Return the field data projected on mesh of this TBROM."""
        return self._meshdata

    @property
    def field_on_points(self):
        """Return the field data on points cloud of this TBROM."""
        return self._pointsdata

    @property
    def nb_points(self):
        """Return the number of points of this TBROM."""
        return self._nbpoints

    @property
    def nb_modes(self):
        """Return the number of modes of this TBROM output field."""
        return self._nbmodes
