import json
import os
import struct

import numpy as np


class TbRom:
    """
    Instantiates a TBROM that is part of a Twin file created by Ansys Twin Builder.

    After a twin model is initialized and its TBROM is instantiated, additional functionalities
    are available to generate snapshots (in memory or on disk) and project input field data
    (if the TBROM is parameterized with an input field).

    Parameters
    ----------
    tbrom_name : str
        Name of the TBROM included in the twin file.
    tbrom_path : str
        File path to the folder with TBROM files.
    """

    IN_F_KEY = "binaryInputField"
    OUT_F_KEY = "binaryOutputField"
    TBROM_BASIS = "basis.svd"
    TBROM_SET = "settings.json"
    TBROM_POINTS = "points.bin"

    def __init__(self, tbrom_name: str, tbrom_path: str):
        self._tbrom_path = tbrom_path
        self._tbrom_name = tbrom_name
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

    def points_generation(self, on_disk: bool, output_file_path: str, namedselection: str = None):
        """
        Generate a point file for the full field or a specific part.

        The point file can be saved in memory or written to disk. When it is written to disk,
        the path specified for the ``output_file_name`` parameter is used.

        Parameters
        ----------
        on_disk : bool
            Whether the point file is saved on disk (True) or returned in memory (False).
        output_file_name: str
            Path for where the point file is written when saved to disk.
        named_selection: str (optional)
            Named selection on which the point file has to be generated
        """
        pointpath = os.path.join(self._tbrom_path, TbRom.OUT_F_KEY, TbRom.TBROM_POINTS)
        vec = np.array(TbRom._read_binary(pointpath))
        if namedselection is not None:
            pointsids = self.namedselectionids(namedselection)
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

    def snapshot_generation(self, on_disk: bool, output_file_path: str, namedselection: str = None):
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
        output_file_name: str
            Path where the snapshot file is written if on_disk is True
        named_selection: str (optional)
            Named selection on which the snasphot has to be generated
        """
        basis = self._outbasis
        vec = np.zeros(len(basis[0]))
        nb_mc = len(basis)
        mc = list(self._outmcs.values())
        for i in range(nb_mc):
            mnp = np.array(basis[i])
            vec = vec + mc[i] * mnp
        if namedselection is not None:
            pointsids = self.namedselectionids(namedselection)
            listids = []
            for i in pointsids:
                for k in range(0, self.outputfielddimensionality):
                    listids.append(i * self.outputfielddimensionality + k)
            vec = vec[listids]
        if on_disk:
            TbRom._write_binary(output_file_path, vec)
            return output_file_path
        else:
            return vec

    def snapshot_projection(self, snapshot: str, fieldname: str = None):
        """
        Project a given snapshot file on the basis associated to the input field name 'fieldname'

        Parameters
        ----------
        snapshot: str
            Path of the input field snapshot file
        fieldname: str (optional)
            Name of the input field for which the snapshot projection will be performed (it needs to be defined in case
            the TBROM is parameterized with multiple input fields)
        """
        mc = []
        vec = TbRom._read_binary(snapshot)
        vecnp = np.array(vec)
        if fieldname is None or self.numberinputfields == 1:
            basis = list(self._infbasis.values())[0]
        else:
            basis = self._infbasis[fieldname]
        nb_mc = len(basis)
        for i in range(nb_mc):
            mnp = np.array(basis[i])
            mci = mnp.dot(vecnp)
            mc.append(mci)
        if fieldname is None or self.numberinputfields == 1:
            index = 0
            for item, key in self._infmcs[self.nameinputfields[0]].items():
                self._infmcs[self.nameinputfields[0]][item] = mc[index]
                index = index + 1
        else:
            index = 0
            for item, key in self._infmcs[fieldname].items():
                self._infmcs[fieldname][item] = mc[index]
                index = index + 1

    def namedselectionids(self, nsname: str):
        return self._nsidslist[nsname]

    def fieldinputmodecoefficients(self, fieldname: str):
        return self._infmcs[fieldname]

    def hasinfmcs(self, fieldname: str):
        return self._hasinfmcs[fieldname]

    def input_field_size(self, fieldname: str):
        return len(self._infbasis[fieldname][0])

    @staticmethod
    def _read_basis(fn):
        fr = open(fn, "rb")
        var = struct.unpack("cccccccccccccccc", fr.read(16))[0]
        nb_val = struct.unpack("Q", fr.read(8))[0]
        nb_mc = struct.unpack("Q", fr.read(8))[0]
        basis = []
        for i in range(nb_mc):
            vec = []
            for j in range(nb_val):
                vec.append(struct.unpack("d", fr.read(8))[0])
            basis.append(vec)
        fr.close()
        return basis

    @staticmethod
    def _read_binary(file):
        fr = open(file, "rb")
        nbdof = struct.unpack("Q", fr.read(8))[0]
        vec = []
        for i in range(nbdof):
            vec.append(struct.unpack("d", fr.read(8))[0])
        fr.close()
        return vec

    @staticmethod
    def _write_binary(fn, vec):
        fw = open(fn, "wb")
        fw.write(struct.pack("Q", len(vec)))
        for i in vec:
            fw.write(struct.pack("d", i))
        return True

    @staticmethod
    def _read_settings(settingspath):
        f = open(settingspath)
        data = json.load(f)
        namedselection = data["namedSelections"]
        dimensionality = data["dimensionality"]
        name = data["name"]
        unit = data["unit"]
        tbromns = dict()
        outputname = name.replace(" ", "_")
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
    def read_snapshot_size(file):
        fr = open(file, "rb")
        nbdof = struct.unpack("Q", fr.read(8))[0]
        return nbdof

    @property
    def outmcs(self):
        return self._outmcs

    @property
    def hasoutmcs(self):
        return self._hasoutmcs

    @property
    def tbromname(self):
        return self._tbrom_name

    @property
    def numberinputfields(self):
        return len(self._infbasis)

    @property
    def nameinputfields(self):
        return list(self._infbasis.keys())

    @property
    def infmcs(self):
        return self._infmcs

    @property
    def nsnames(self):
        return list(self._nsidslist.keys())

    @property
    def outputfieldname(self):
        return self._outname

    @property
    def outputfieldunit(self):
        return self._outunit

    @property
    def outputfielddimensionality(self):
        return self._outdim
