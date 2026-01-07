# Copyright (C) 2022 - 2026 ANSYS, Inc. and/or its affiliates.
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

import numpy as np
from pytwin import (
    stress_strain_component,
)


def allclose_and_same_shape(a: np.ndarray, b: np.ndarray, rtol=1e-05, atol=1e-08):
    """Compare arrays for identical shape and entries within tolerance."""
    return np.allclose(a, b, rtol=rtol, atol=atol) and (a.shape == b.shape)


def almost_collinear_unit_vectors(vectors: np.ndarray, reference: np.ndarray, rtol=1e-05, atol=1e-08, axis=1):
    """Check if vectors are almost collinear with reference and have unit length."""
    all_collinear = np.allclose(np.cross(vectors, reference, axis=axis), np.zeros_like(reference), rtol=rtol, atol=atol)
    all_unit = np.allclose(np.linalg.norm(vectors, axis=axis), np.ones(vectors.shape[0]))
    return all_collinear and all_unit


class TestPostprocessing:

    def test_stress_strain_scalar_output(self):

        # Reference inputs from Ansys Mechanical
        pr = 0.3
        stress_vectors = np.array(
            [
                [168.123993, 533.840027, 149.458389, -18.4201412, 84.3208466, 19.6648216],
                [-14.2597857, -47.3646698, -15.0713329, 115.835953, -49.9352531, 19.6648216],
            ]
        )
        strain_vectors = np.array(
            [
                [
                    -1.843276987e-04,
                    2.192826709e-03,
                    -3.056541027e-04,
                    -2.394618350e-04,
                    1.096170978e-03,
                    2.556426916e-04,
                ],
                [
                    2.235507964e-05,
                    -1.928266720e-04,
                    1.708001764e-05,
                    1.505867462e-03,
                    -6.491582608e-04,
                    2.556426916e-04,
                ],
            ]
        )
        # Reference outputs (from Ansys Mechanical)
        principal_directions = np.array(
            [
                [
                    [-0.03654073, 0.97846409, 0.20315707],
                    [0.89781754, -0.05713118, 0.43664596],
                    [-0.438849, -0.19835334, 0.87639461],
                ],
                [
                    [0.71548582, 0.67320539, -0.18674727],
                    [0.35315995, -0.11788318, 0.92810646],
                    [-0.60279191, 0.72999867, 0.32209292],
                ],
            ]
        )
        principal_stress = np.array([[552.035341, 178.859949, 120.52712], [89.5983668, -1.24602911, -165.048126]])
        principal_strain = np.array(
            [
                [2.311096237e-03, -1.145439771e-04, -4.937073526e-04],
                [6.974330997e-04, 1.069444900e-04, -9.577691644e-04],
            ]
        )
        stress_int = np.array([431.508221, 254.646493])
        strain_int = np.array([2.804803590e-03, 1.655202264e-03])
        eqv_stress = np.array([405.500886, 223.527031])
        eqv_strain = np.array([2.027504612e-03, 1.117635169e-03])
        maxShear_stress = np.array([215.7541105, 127.3232464])
        maxShear_strain = np.array([2.804803590e-03, 1.655202264e-03])
        absMaxPrin_stress = np.array([552.035341, -165.048126])
        sgnEqv_stress = np.array([405.500886, -223.527031])
        for idx, comp in enumerate(["X", "Y", "Z", "XY", "YZ", "XZ"]):
            assert np.array_equal(stress_strain_component(stress_vectors, "S", comp), stress_vectors[:, idx])
            assert np.array_equal(stress_strain_component(strain_vectors, "E", comp), strain_vectors[:, idx])
        for idx, comp in enumerate(["dir1", "dir2", "dir3"]):
            stress = stress_strain_component(stress_vectors, "S", comp)
            strain = stress_strain_component(strain_vectors, "E", comp)
            assert almost_collinear_unit_vectors(stress, principal_directions[:, idx], axis=1)
            assert almost_collinear_unit_vectors(strain, principal_directions[:, idx], axis=1, atol=1e-7)
        for idx, comp in enumerate([1, 2, 3]):
            stress = stress_strain_component(stress_vectors, "S", comp)
            strain = stress_strain_component(strain_vectors, "E", comp)
            assert allclose_and_same_shape(stress, principal_stress[:, idx])
            assert allclose_and_same_shape(strain, principal_strain[:, idx], atol=1e-11)
        assert allclose_and_same_shape(stress_strain_component(stress_vectors, "S", "INT"), stress_int)
        assert allclose_and_same_shape(stress_strain_component(strain_vectors, "E", "INT"), strain_int, atol=1e-11)
        assert allclose_and_same_shape(stress_strain_component(stress_vectors, "S", "EQV"), eqv_stress)
        assert allclose_and_same_shape(
            stress_strain_component(strain_vectors, "E", "EQV", effective_pr=pr), eqv_strain, atol=1e-11
        )
        assert allclose_and_same_shape(stress_strain_component(stress_vectors, "S", "maxShear"), maxShear_stress)
        assert allclose_and_same_shape(
            stress_strain_component(strain_vectors, "E", "maxShear"), maxShear_strain, atol=1e-11
        )
        assert allclose_and_same_shape(stress_strain_component(stress_vectors, "S", "sgnEQV"), sgnEqv_stress)
        assert allclose_and_same_shape(stress_strain_component(stress_vectors, "S", "absMaxPrin"), absMaxPrin_stress)

    def test_stress_strain_scalar_output_errors(self):
        stress_vectors = np.array(
            [
                [168.123993, 533.840027, 149.458389, -18.4201412, 84.3208466, 19.6648216],
                [-14.2597857, -47.3646698, -15.0713329, 115.835953, -49.9352531, 19.6648216],
            ]
        )
        try:
            output = stress_strain_component(stress_vectors[:, :5], "S", "X")
        except ValueError as e:
            assert (
                f"Input array shape is {stress_vectors[:, :5].shape}, but must be {stress_vectors.shape[0], 6}."
                in str(e)
            )
        try:
            output = stress_strain_component(stress_vectors, "S", "noComp")
        except ValueError as e:
            assert "Invalid component 'noComp' for item 'S.'" in str(e)
        try:
            output = stress_strain_component(stress_vectors, "E", "absMaxPrin")
        except ValueError as e:
            assert "Invalid component 'absMaxPrin' for item 'E.'" in str(e)
        try:
            output = stress_strain_component(stress_vectors, "E", "EQV")
        except ValueError as e:
            assert "Enter a valid effective Poisson's ratio to calculate equivalent strain." in str(e)
        try:
            output = stress_strain_component(stress_vectors, "X", "EQV")
        except ValueError as e:
            assert "Invalid 'item' label, 'X'. Valid labels are 'S' and 'E'." in str(e)
