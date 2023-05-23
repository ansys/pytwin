import os
import struct

import numpy as np

# method to update
# twin input mode ceof update based on projection
# tbrom output mode coef based on Twin Outputs -> this needs to be called whenver twin outputs are updated (to do batch mode) -> to discuss
# tbrom functionalities should be available based on hasInputMode/hasOutputModeCoef -> twin_model methods to expose snapshot generaton/projection
# treat corner cases when no tbrom, or no output mode coef are exposed and/or input mode coef (or with different names than ones expected)
# optional arguemtn to output snapshot on specified parts (ASML workflow)
# fix workaround about boolean hasOutputModeCoef check
# corner cases with multiple input fields and even multiple tbrom

class TbRom():
    def __init__(self, tbrom_name: str, tbrom_path: str):
        self._tbrom_path = tbrom_path
        self._tbrom_name = tbrom_name
        self._inputModeCoefficients = None
        self._outputModeCoefficients = None
        self._inputSVD = []
        self._outputSVD = None
        self._inputFieldNames = []
        self._hasInputModeCoefficients = False  # this will indicate whether the upper Twin has inputs connected to
        # TBROM input mode coefficients (if any input field)
        self._hasOutputModeCoefficients = False  # this will indicate whether the upper Twin has outputs connected to
        # TBROM output mode coefficient (if any defined)

        # based on tbrom_path, find inputSVD (if any) and outputSVD
        files = os.listdir(tbrom_path)
        for file in files:
            if 'binaryInputField' in file:
                folder = file.split('_')
                self._inputFieldNames.append(folder[1])
                inputSVDpath = os.path.join(tbrom_path, file, "basis.svd")
                self._inputSVD.append(TbRom._read_basis(inputSVDpath))

        outputSVDpath = os.path.join(tbrom_path, "binaryOutputField", "basis.svd")
        self._outputSVD = TbRom._read_basis(outputSVDpath)

    def snapshot_projection(self, snapshot):
        if self.hasInputModeCoefficients:
            modes_coef = []
            vec = TbRom._read_binary(snapshot)
            vecnp = np.array(vec)
            basis = self._inputSVD
            nb_modes = len(basis)
            for i in range(nb_modes):
                modenp = np.array(basis[i])
                coef = modenp.dot(vecnp)
                modes_coef.append(coef)
            for item, key in self._inputModeCoefficients.items():
                self._inputModeCoefficients[item] = modes_coef
        else:
            return []

    def snapshot_generation(self, on_disk, output_file):
        if self.hasOutputModeCoefficients:
            basis = self._outputSVD
            vec = np.zeros(len(basis[0]))
            nb_modes = len(basis)
            modes_coefs = list(self._outputModeCoefficients.values())
            for i in range(nb_modes):
                modenp = np.array(basis[i])
                vec = vec + modes_coefs[i] * modenp
            if on_disk:
                TbRom._write_binary(output_file, vec)
            else:
                return vec
        else:
            return []

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

    @property
    def outputModeCoefficients(self):
        return self._outputModeCoefficients

    @property
    def inputModeCoefficients(self):
        return self._inputModeCoefficients

    @property
    def hasInputModeCoefficients(self):
        return self._hasInputModeCoefficients

    @property
    def hasOutputModeCoefficients(self):
        return self._hasOutputModeCoefficients

    @property
    def TbRomName(self):
        return self._tbrom_name
