import json
import os
from pathlib import Path
import struct
from typing import Union

import numpy as np
import pyvista as pv


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
        self._hasinfmcs = None
        self._hasoutmcs = False

        files = os.listdir(tbrom_path)
        infdata = dict()

        for file in files:
            if TbRom.IN_F_KEY in file:
                folder = file.split("_")
                fname = folder[1]
                inpath = os.path.join(tbrom_path, file, TbRom.TBROM_BASIS)
                inbasis = TbRom._read_basis(inpath)
                infdata.update({fname: inbasis})
        self._infbasis = infdata

        settingspath = os.path.join(tbrom_path, TbRom.OUT_F_KEY, TbRom.TBROM_SET)
        [nsidslist, dimensionality, outputname, unit] = TbRom._read_settings(settingspath)
        self._nsidslist = nsidslist
        self._outdim = int(dimensionality[0])
        self._outname = outputname
        self._outunit = unit
        self._outputfilespath = None
        propertiespath = os.path.join(tbrom_path, TbRom.TBROM_PROP)
        [nbpoints, nbmodes] = TbRom._read_properties(propertiespath)
        self._nbpoints = int(nbpoints / self._outdim)
        self._nbmodes = nbmodes

        pointpath = os.path.join(tbrom_path, TbRom.OUT_F_KEY, TbRom.TBROM_POINTS)
        if not os.path.exists(pointpath):
            self._haspointfile = False
        else:
            self._haspointfile = True
        self._read_points(pointpath)

        outpath = os.path.join(tbrom_path, TbRom.OUT_F_KEY, TbRom.TBROM_BASIS)
        self._init_pointsdata(outpath)

    def generate_points(self, on_disk: bool, output_file_path: str, named_selection: str = None):
        """
        Generate a point file for the full field or a specific part.

        The point file can be saved in memory or written to disk. When it is written to disk,
        the path specified for the ``output_file_name`` parameter is used.

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
        vec = self.data_extract(named_selection, self._pointsdata.points)
        if on_disk:
            TbRom._write_binary(output_file_path, vec)
            return output_file_path
        else:
            return vec

    def generate_snapshot(self, on_disk: bool, output_file_path: str, named_selection: str = None):
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
        vec = self.data_extract(named_selection, self._pointsdata[self.field_output_name])
        if on_disk:
            TbRom._write_binary(output_file_path, vec)
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
            vecnp = TbRom._read_binary(snapshot)

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

    def update_output_field(self):
        """
        Compute the output field results with current mode coefficients.
        """
        mc = list(self._outmcs.values())
        self._pointsdata[self.field_output_name] = mc[0] * self._pointsdata["mode" + str(1)]
        for i in range(1, len(mc)):
            self._pointsdata[self.field_output_name] = (
                self._pointsdata[self.field_output_name] + mc[i] * self._pointsdata["mode" + str(i + 1)]
            )
        self._pointsdata.set_active_scalars(self.field_output_name)

    def named_selection_indexes(self, nsname: str):
        return self._nsidslist[nsname]

    def input_field_size(self, fieldname: str):
        return len(self._infbasis[fieldname][0])

    def data_extract(self, named_selection: str, data: np.ndarray):
        if named_selection is not None:
            pointsids = self.named_selection_indexes(named_selection)
            listids = np.sort(pointsids)
            return data[listids]
        else:
            return data

    def _read_points(self, filepath):
        if self.haspointfile:
            points = TbRom._read_binary(filepath)
        else:
            points = np.zeros(3 * self.nb_points)
        self._pointsdata = pv.PolyData(points.reshape(-1, 3))

    def _init_pointsdata(self, filepath):
        basis = TbRom._read_basis(filepath)
        for i in range(0, len(basis)):
            self._pointsdata["mode" + str(i + 1)] = basis[i].reshape(-1, self.field_output_dim)
        # initialize output field data
        if self._hasoutmcs:
            self._pointsdata[self.field_output_name] = np.zeros((self.nb_points, self.field_output_dim))

    @staticmethod
    def _read_basis(filepath):
        with open(filepath, "rb") as f:
            var = struct.unpack("cccccccccccccccc", f.read(16))[0]
            nb_val = struct.unpack("Q", f.read(8))[0]
            nb_mc = struct.unpack("Q", f.read(8))[0]
            return np.fromfile(f, dtype=np.double, offset=0).reshape(-1, nb_val)

    @staticmethod
    def _read_binary(filepath):
        return np.fromfile(filepath, dtype=np.double, offset=8).reshape(
            -1,
        )

    @staticmethod
    def _write_binary(filepath, vec):
        vec = vec.reshape(
            -1,
        )
        if os.path.exists(filepath):
            os.remove(filepath)
        with open(filepath, "xb") as f:
            f.write(struct.pack("Q", len(vec)))
            vec.tofile(f)
        return True

    @staticmethod
    def _read_settings(filepath):
        with open(filepath) as f:
            data = json.load(f)

        namedselection = {}
        dimensionality = None
        name = None
        unit = None

        if "namedSelections" in data:
            namedselection = data["namedSelections"]
        if "dimensionality" in data:
            dimensionality = data["dimensionality"]
        if "name" in data:
            name = data["name"]
        if "unit" in data:
            unit = data["unit"]

        tbromns = dict()
        outputname = name.replace(" ", "_")

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

    @staticmethod
    def _read_properties(filepath):
        with open(filepath) as f:
            data = json.load(f)

        fields = {}
        if "fields" in data:
            fields = data["fields"]
        outField = fields["outField"]

        nbPoints = outField["nbDof"]
        nbModes = outField["nbModes"]

        return [nbPoints, nbModes]

    @staticmethod
    def read_snapshot_size(filepath):
        with open(filepath, "rb") as f:
            nbdof = struct.unpack("Q", f.read(8))[0]
        return nbdof

    @property
    def haspointfile(self):
        return self._haspointfile

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
