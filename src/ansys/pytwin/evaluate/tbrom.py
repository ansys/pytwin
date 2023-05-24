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
# 1 Twin with no TBROMs (new implementation should work on existing examples)
# 1 Twin with multiple TBROMs with all mode coef connected as Twin inputs/outputs
# - without input field
# - with 1 input field
# - with mulitple input fields
# 1 Twin with multiple TBROMs with 1 TBROM having inputs/outputs not connected or wrong names exposed (existing examples)
# same as above
# snapshot_projection and snapshot_generation
# - what about wrong arguments passed
# - arguemtns are not consistent (e.g. input snapshot not consistent with input field SVD)
# - return output if function not applicable



class TbRom():
    def __init__(self, tbrom_name: str, tbrom_path: str):
        self._tbrom_path = tbrom_path
        self._tbrom_name = tbrom_name
        self._inputFieldsModeCoefficients = None
        self._outputModeCoefficients = None
        self._inputFieldsSVD = None
        self._outputSVD = None
        self._hasInputModeCoefficients = False  # this will indicate whether the upper Twin has inputs connected to
        # TBROM input mode coefficients (if any input field)
        self._hasOutputModeCoefficients = False  # this will indicate whether the upper Twin has outputs connected to
        # TBROM output mode coefficient (if any defined)

        # based on tbrom_path, find inputSVD (if any) and outputSVD
        files = os.listdir(tbrom_path)
        inputFieldData = dict()
        for file in files:
            if 'binaryInputField' in file:
                folder = file.split('_')
                fieldName = folder[1]
                inputSVDpath = os.path.join(tbrom_path, file, "basis.svd")
                svd_basis = TbRom._read_basis(inputSVDpath)
                inputFieldData.update({fieldName: svd_basis})
        self._inputFieldsSVD = inputFieldData

        outputSVDpath = os.path.join(tbrom_path, "binaryOutputField", "basis.svd")
        self._outputSVD = TbRom._read_basis(outputSVDpath)
        settingsPath = os.path.join(tbrom_path, "binaryOutputField", "settings.json")
        self._NsIdsList = TbRom._read_settings(settingsPath)

    def snapshot_projection(self, snapshot: str, fieldname: str = None):
        if self.hasInputModeCoefficients:
            modes_coef = []
            vec = TbRom._read_binary(snapshot)
            vecnp = np.array(vec)
            if fieldname is None or self.NumberInputField == 1:
                basis = list(self._inputFieldsSVD.values())[0]
            else:
                basis = self._inputFieldsSVD[fieldname]
            nb_modes = len(basis)
            for i in range(nb_modes):
                modenp = np.array(basis[i])
                coef = modenp.dot(vecnp)
                modes_coef.append(coef)
            if fieldname is None or self.NumberInputField == 1:
                index = 0
                for item, key in self._inputFieldsModeCoefficients[self.NameInputFields[0]].items():
                    self._inputFieldsModeCoefficients[self.NameInputFields[0]][item] = modes_coef[index]
                    index = index + 1
            else:
                index = 0
                for item, key in self._inputFieldsModeCoefficients[fieldname].items():
                    self._inputFieldsModeCoefficients[fieldname][item] = modes_coef[index]
                    index = index + 1
        else:
            return []

    def snapshot_generation(self, on_disk, output_file, NamedSelection: str = None):
        if self.hasOutputModeCoefficients:
            basis = self._outputSVD
            vec = np.zeros(len(basis[0]))
            nb_modes = len(basis)
            modes_coefs = list(self._outputModeCoefficients.values())
            for i in range(nb_modes):
                modenp = np.array(basis[i])
                vec = vec + modes_coefs[i] * modenp
            if NamedSelection is not None:
                vec = vec[self.namedselectionids(NamedSelection)]
            if on_disk:
                TbRom._write_binary(output_file, vec)
            else:
                return vec
        else:
            return []

    def namedselectionids(self, NsName: str):
        return self._NsIdsList[NsName]

    def fieldinputmodecoefficients(self, fieldName: str):
        return self._inputFieldsModeCoefficients[fieldName]

    @staticmethod
    def _read_basis(fn):
        fr = open(fn, "rb")
        var = struct.unpack('cccccccccccccccc', fr.read(16))[0]
        nb_val = struct.unpack('Q', fr.read(8))[0]
        nb_modes = struct.unpack('Q', fr.read(8))[0]
        basis = []
        for i in range(nb_modes):
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
        for i in vec:
            fw.write(struct.pack("d", i))
        return True

    @staticmethod
    def _read_settings(settingsPath):
        f = open(settingsPath)
        data = json.load(f)
        namedSelection = data['namedSelections']
        tbRomNS = dict()
        for name, idsList in namedSelection.items():
            finalList = []
            for i in range(0,len(idsList)-1):
                if int(idsList[i])==-1:
                    for j in range(int(idsList[i-1])+1,int(idsList[i+1])):
                        finalList.append(j)
                else:
                    finalList.append(int(idsList[i]))
            finalList.append(int(idsList[len(idsList)-1]))
            tbRomNS.update({name:finalList})
        return tbRomNS


    @property
    def outputModeCoefficients(self):
        return self._outputModeCoefficients

    @property
    def hasInputModeCoefficients(self):
        return self._hasInputModeCoefficients

    @property
    def hasOutputModeCoefficients(self):
        return self._hasOutputModeCoefficients

    @property
    def TbRomName(self):
        return self._tbrom_name

    @property
    def NumberInputField(self):
        return len(self._inputFieldsSVD)

    @property
    def NameInputFields(self):
        return list(self._inputFieldsSVD.keys())

    @property
    def NamedSelectionNames(self):
        return list(self._NsIdsList.keys())




