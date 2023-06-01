import json
import os
import struct

import numpy as np


# need a clear convention as long as we don't have direct apis
# for example :
# - outputs :
# if 1 single TBROM: that's fine as long as it contains _mode_
# if >0 TBROM: twin output name needs to contain tbrom_name
# - inputs :
# if >0 single TBROM: it contains name of field + _mode_
# if >0 : needs to contain tbrom_name

# input field = dict of {rom_name, {field_name, snapshot file path}}

# corner cases to handle/test
# snapshot_projection and snapshot_generation
# - what about wrong arguments passed
# - arguemtns are not consistent (e.g. input snapshot not consistent with input field SVD)
# - return output if function not applicable
# naming obstrufaction
# unit testing
# - check if given twin with tbrom has input/output mode coef
#    - ok for 1 tbrom
#    - test same with multiple tbrom
# - check if ns selection ids returned are correct
# - check if input mode coef returned are correct
# - check snapshot generation method
# - check twin initialization with tbrom / without tbrom -> OK
# - check twin outputs update / inputs update
# _attributes vs public
# documentation


class TbRom:
    """
    Instantiates a TBROM model part of a TWIN file created by Ansys Twin Builder.

    After a twin model is initialized and its tbrom are instantiated, additional functionalities are available to
    generate snapshots (in memory or on disk), as well as to project input field data (in case the tbrom is
    parameterized with an input field).

    Parameters
    ----------
    tbrom_name : str
        Name of the TBROM included in the Twin.
    tbrom_path : str
        File path to the TBROM files folder.
    """

    IN_F_KEY = 'binaryInputField'
    OUT_F_KEY = 'binaryOutputField'
    TBROM_BASIS = 'basis.svd'
    TBROM_SET = 'settings.json'

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
                folder = file.split('_')
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

    def snapshot_generation(self, on_disk, output_file_name, namedselection: str = None):
        if self.hasoutmcs:
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
                TbRom._write_binary(os.path.join(self._outputfilespath, output_file_name), vec)
            else:
                return vec
        else:
            return []

    def snapshot_projection(self, snapshot: str, fieldname: str = None):
        if self._hasinfmcs[fieldname]:
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
        else:
            return []

    def namedselectionids(self, nsname: str):
        return self._nsidslist[nsname]

    def fieldinputmodecoefficients(self, fieldname: str):
        return self._infmcs[fieldname]

    def hasinfmcs(self, fieldname: str):
        return self._hasinfmcs[fieldname]

    def _points_generation(self, on_disk, output_file_name, namedselection):
        pointpath = os.path.join(self._tbrom_path, TbRom.OUT_F_KEY, "points.bin")
        points = np.array(TbRom._read_binary(pointpath))
        pointsids = self.namedselectionids(namedselection)
        listids = []
        for i in pointsids:
            for k in range(0, 3):
                listids.append(i * 3 + k)
        vec = points[listids]
        if on_disk:
            TbRom._write_binary(os.path.join(self._outputfilespath, output_file_name), vec)
        else:
            return vec

    @staticmethod
    def _read_basis(fn):
        fr = open(fn, "rb")
        var = struct.unpack('cccccccccccccccc', fr.read(16))[0]
        nb_val = struct.unpack('Q', fr.read(8))[0]
        nb_mc = struct.unpack('Q', fr.read(8))[0]
        basis = []
        for i in range(nb_mc):
            vec = []
            for j in range(nb_val):
                vec.append(struct.unpack('d', fr.read(8))[0])
            basis.append(vec)
        fr.close()
        return basis

    @staticmethod
    def _read_binary(file):
        fr = open(file, "rb")
        nbdof = struct.unpack('Q', fr.read(8))[0]
        vec = []
        for i in range(nbdof):
            vec.append(struct.unpack('d', fr.read(8))[0])
        fr.close()
        return vec

    @staticmethod
    def _write_binary(fn, vec):
        fw = open(fn, "wb")
        fw.write(struct.pack("Q", len(vec)))
        print(fn)
        for i in vec:
            fw.write(struct.pack("d", i))
        return True

    @staticmethod
    def _read_settings(settingspath):
        f = open(settingspath)
        data = json.load(f)
        namedselection = data['namedSelections']
        dimensionality = data['dimensionality']
        name = data['name']
        unit = data['unit']
        tbromns = dict()
        outputname = name.replace(' ', '_')
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
