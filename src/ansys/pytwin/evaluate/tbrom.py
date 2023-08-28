import json
import os
import struct

import numpy as np


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
    TBROM_POINTS = "points.bin"

    def __init__(self, tbrom_name: str, tbrom_path: str):
        self._tbrom_path = tbrom_path
        self._name = tbrom_name
        self._infmcs = None
        self._outmcs = None
        self._infbasis = None
        self._outbasis = None
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

        outpath = os.path.join(tbrom_path, TbRom.OUT_F_KEY, TbRom.TBROM_BASIS)
        self._outbasis = TbRom._read_basis(outpath)
        settingspath = os.path.join(tbrom_path, TbRom.OUT_F_KEY, TbRom.TBROM_SET)
        [nsidslist, dimensionality, outputname, unit] = TbRom._read_settings(settingspath)
        self._nsidslist = nsidslist
        self._outdim = int(dimensionality[0])
        self._outname = outputname
        self._outunit = unit
        self._outputfilespath = None

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
        pointpath = os.path.join(self._tbrom_path, TbRom.OUT_F_KEY, TbRom.TBROM_POINTS)
        vec = np.array(TbRom._read_binary(pointpath))
        if named_selection is not None:
            pointsids = self.named_selection_indexes(named_selection)
            listids = []
            for i in pointsids:
                for k in range(0, 3):
                    listids.append(i * 3 + k)
            vec = vec[listids]
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
        basis = self._outbasis
        vec = np.zeros(len(basis[0]))
        nb_mc = len(basis)
        mc = list(self._outmcs.values())
        for i in range(nb_mc):
            mnp = np.array(basis[i])
            vec = vec + mc[i] * mnp
        if named_selection is not None:
            pointsids = self.named_selection_indexes(named_selection)
            listids = []
            for i in pointsids:
                for k in range(0, self.field_output_dim):
                    listids.append(i * self.field_output_dim + k)
            vec = vec[listids]
        if on_disk:
            TbRom._write_binary(output_file_path, vec)
            return output_file_path
        else:
            return vec

    def _reduce_field_input(self, name: str, snapshot_filepath: str):
        """
        Project a snapshot file associated to the input field name ``fieldname``

        Parameters
        ----------
        snapshot_filepath: str
            Path of the input field snapshot file
        name: str (optional)
            Name of the input field to project the snapshot. The name of the field must be specified in case the TBROM
            is parameterized with multiple input fields.
        """
        mc = []
        vec = TbRom._read_binary(snapshot_filepath)
        vecnp = np.array(vec)
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

    def named_selection_indexes(self, nsname: str):
        return self._nsidslist[nsname]

    def input_field_size(self, fieldname: str):
        return len(self._infbasis[fieldname][0])

    @staticmethod
    def _read_basis(filepath):
        with open(filepath, "rb") as f:
            var = struct.unpack("cccccccccccccccc", f.read(16))[0]
            nb_val = struct.unpack("Q", f.read(8))[0]
            nb_mc = struct.unpack("Q", f.read(8))[0]
            basis = []
            for i in range(nb_mc):
                vec = []
                for j in range(nb_val):
                    vec.append(struct.unpack("d", f.read(8))[0])
                basis.append(vec)
        return basis

    @staticmethod
    def _read_binary(filepath):
        with open(filepath, "rb") as f:
            nbdof = struct.unpack("Q", f.read(8))[0]
            vec = []
            for i in range(nbdof):
                vec.append(struct.unpack("d", f.read(8))[0])
        return vec

    @staticmethod
    def _write_binary(filepath, vec):
        if os.path.exists(filepath):
            os.remove(filepath)
        with open(filepath, "xb") as f:
            f.write(struct.pack("Q", len(vec)))
            for i in vec:
                f.write(struct.pack("d", i))
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
            finallist = []
            for i in range(0, len(idsList) - 1):
                if int(idsList[i]) == -1:
                    for j in range(int(idsList[i - 1]) + 1, int(idsList[i + 1])):
                        finallist.append(j)
                else:
                    finallist.append(int(idsList[i]))
            finallist.append(int(idsList[len(idsList) - 1]))
            tbromns.update({name: finallist})

        return [tbromns, dimensionality, outputname, unit]

    @staticmethod
    def read_snapshot_size(filepath):
        with open(filepath, "rb") as f:
            nbdof = struct.unpack("Q", f.read(8))[0]
        return nbdof

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
