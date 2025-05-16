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

import numpy as np

#: Map tensor components to TBROM snapshot ordering
TENSOR_MAP = {"X": 0, "Y": 1, "Z": 2, "XY": 3, "YZ": 4, "XZ": 5}


def stress_strain_component(
    str_vectors: np.ndarray,
    item: str,
    comp: str | int,
    effective_pr: float | None = None,
):
    """
    Reduce an array of stress or strain tensors to an array of scalar values.

    The function applies transformations to return the following outputs at each input location from an input snapshot
    stress or strain vector:

    - Individual stress or strain normal or shear components.
    - Individual stress or strain principal components.
    - Stress or strain intensity.
    - Equivalent (von Mises) stress or strain.
    - Maximum shear stress or strain.
    - Absolute maximum principal stress.
    - Signed equivalent (von Mises) stress.

    Parameters
    ----------
    str_vectors : np.ndarray
        ``(n, 6)`` array of stress or strain vectors of symmetric Cauchy tensor components, X,Y,Z,XY,YZ,XZ.

        This ordering is consistent with the conventions used in Ansys TBROM snapshots and Ansys Mechanical APDL
        (`2.1.1. Stress-Strain Relationships`_).

        Shear strains are interpreted as being the engineering shear strains, which are twice the tensor shear strains.

        Planar stress or strain values can be entered by setting the relevant out-of-plane values to zero.
    item : str
        Label identifying the result type of ``str_vectors``. ``S`` for stress and ``E`` for strain.
    comp : str | int
        Component of the item. See the table below in the notes section.
    effective_pr : float | None, default = None
        Effective Poisson's ratio for calculating equivalent strain. Assumed to be constant for all entries in
        ``str_vectors``. Refer to `2.4. Combined Stresses and Strains`_ for potential values when handling strains other
        than elastic strain.

    Returns
    -------
    (n,) array | (n,3) array
        ``(n,)`` array of the requested scalar stress or strain values, or ``(n, 3)`` array of principal component
        direction vectors when ``dir1, dir2, dir3`` are selected as the output component.

    Raises
    ------
    ValueError
        if shape of ``str_vectors`` is not ``(n, 6)``.
    ValueError
        if invalid combinations for ``item`` and ``comp`` are entered.
    ValueError
        if ``item = E`` and ``comp = EQV`` and ``effective_pr`` is not given.

    Notes
    -----
    This table lists the results values available to this method.

    +------+---------------------+--------------------------------------+
    | item | comp                | Description                          |
    +------+---------------------+--------------------------------------+
    | S    | X, Y, Z, XY, YZ, XZ | Component stress.                    |
    |      +---------------------+--------------------------------------+
    |      | 1, 2, 3             | Principal stress.                    |
    |      +---------------------+--------------------------------------+
    |      | dir1, dir2, dir3    | Principal stress direction cosine.   |
    |      +---------------------+--------------------------------------+
    |      | INT                 | Stress intensity.                    |
    |      +---------------------+--------------------------------------+
    |      | EQV                 | Equivalent (von Mises) stress.       |
    |      +---------------------+--------------------------------------+
    |      | maxShear            | Maximum shear stress                 |
    |      +---------------------+--------------------------------------+
    |      | absMaxPrin          | Absolute maximum principal stress.   |
    |      +---------------------+--------------------------------------+
    |      | sgnEQV              | Signed equivalent (von Mises) stress.|
    +------+---------------------+--------------------------------------+
    | E    | X, Y, Z, XY, YZ, XZ | Component strain.                    |
    |      +---------------------+--------------------------------------+
    |      | 1, 2, 3             | Principal strain.                    |
    |      +---------------------+--------------------------------------+
    |      | dir1, dir2, dir3    | Principal strain direction cosine.   |
    |      +---------------------+--------------------------------------+
    |      | INT                 | Strain intensity.                    |
    |      +---------------------+--------------------------------------+
    |      | EQV                 | Equivalent (von Mises) strain.       |
    |      +---------------------+--------------------------------------+
    |      | maxShear            | Maximum shear strain                 |
    +------+---------------------+--------------------------------------+

    Outputs are calculated as described in `2.4. Combined Stresses and Strains`_ and `19.5.2.3. Maximum Shear`_ in the
    Ansys Mechanical APDL and Ansys Mechanical help.

    .. _2.1.1. Stress-Strain Relationships: https://ansyshelp.ansys.com/public/account/secured?returnurl=/Views/\
        Secured/corp/v251/en/ans_thry/thy_str1.html%23strucstressstrain
    .. _2.4. Combined Stresses and Strains: https://ansyshelp.ansys.com/public/account/secured?returnurl=/Views/\
        Secured/corp/v251/en/ans_thry/thy_str4.html%23struccombstrain
    .. _19.5.2.3. Maximum Shear: https://ansyshelp.ansys.com/public/account/secured?returnurl=/Views/Secured/corp/\
        v251/en/wb_sim/ds_Maximum_Stress.html

    Examples
    --------
    >>> import numpy as np
    >>> from pytwin import stress_strain_component
    >>> stress_vectors = np.array([[-10,50,0,40,0,0], [15,40,0,30,0,0]])
    >>> S_absMaxPrin = stress_strain_component(stress_vectors, 'S', 'absMaxPrin')
    >>> S_absMaxPrin
    array([70., 60.])

    Calculate equivalent strain for a material with Poisson's ratio of 0.3 (for example steel). Input strains lie in the
    XZ plane.

    >>> strain_vectors = np.array([[-2.1e-4, 0.0, 5.0e-5, 0.0, 0.0, 1.5e-6]])
    >>> E_vonMises = stress_strain_component(strain_vectors, 'E', 'EQV', effective_pr=0.3)
    >>> E_vonMises
    array([0.00018382])


    """
    if str_vectors.shape != (str_vectors.shape[0], 6):
        raise ValueError(f"Input array shape is {str_vectors.shape}, but must be {(str_vectors.shape[0], 6)}.")

    if item == "E" and comp == "EQV":
        try:
            effective_pr = float(effective_pr)
        except TypeError:
            raise ValueError(f"Enter a valid effective Poisson's ratio to calculate equivalent strain.")

    match item, str(comp):
        case "S" | "E", "X" | "Y" | "Z" | "XY" | "YZ" | "XZ":
            # Normal stress/strain components are returned directly
            return str_vectors[:, TENSOR_MAP[comp]].copy()
        case "S", "dir1" | "dir2" | "dir3":
            # Directions are not reused, so return after calculating
            _, directions = _principal_stress_strain(str_vectors)
            idx = int(comp.strip("dir")) - 1
            return directions[:, :, idx]
        case "E", "dir1" | "dir2" | "dir3":
            _, directions = _principal_stress_strain(str_vectors, engineering_strain=True)
            idx = int(comp.strip("dir")) - 1
            return directions[:, :, idx]
        case "S", comp:
            principals, _ = _principal_stress_strain(str_vectors)
        case "E", comp:
            principals, _ = _principal_stress_strain(str_vectors, engineering_strain=True)
        case _:
            raise ValueError(f"Invalid 'item' label, '{item}'. Valid labels are 'S' and 'E'.")

    # Principal values for use in all following calculations
    P1, P2, P3 = principals[:, 0], principals[:, 1], principals[:, 2]

    match item, str(comp):
        case "S" | "E", "1" | "2" | "3":
            idx = int(comp) - 1
            return principals[:, idx]
        case "S" | "E", "INT":
            return np.max(np.stack([np.abs(P1 - P2), np.abs(P2 - P3), np.abs(P3 - P1)], axis=1), axis=1)
        case "S", "EQV":
            return ((1 / 2) * ((P1 - P2) ** 2 + (P2 - P3) ** 2 + (P3 - P1) ** 2)) ** 0.5
        case "E", "EQV":
            eqv_scaling = 1 / (1 + effective_pr)
            return eqv_scaling * ((1 / 2) * ((P1 - P2) ** 2 + (P2 - P3) ** 2 + (P3 - P1) ** 2)) ** 0.5
        case "S", "maxShear":
            return 0.5 * (P1 - P3)
        case "E", "maxShear":
            return P1 - P3
        case "S", "sgnEQV":
            eqv = ((1 / 2) * ((P1 - P2) ** 2 + (P2 - P3) ** 2 + (P3 - P1) ** 2)) ** 0.5
            absMax_indices = np.argmax(np.abs(principals), axis=1, keepdims=True)
            sign = np.sign(np.take_along_axis(principals, absMax_indices, axis=1)).flatten()
            return sign * eqv
        case "S", "absMaxPrin":
            absMax_indices = np.argmax(np.abs(principals), axis=1, keepdims=True)
            return np.take_along_axis(principals, absMax_indices, axis=1).flatten()
        case _:
            raise ValueError(f"Invalid component '{comp}' for item '{item}.'")


def _principal_stress_strain(str_vectors: np.ndarray, engineering_strain: bool = False):
    """
    Compute principal values and direction unit vectors for an array of stress or strain vectors.

    The function transforms vectors of six stress components into symmetrical tensors and then calculates the principal
    components (eigenvalues) and direction vectors (eigenvectors).

    Parameters
    ----------
    str_vectors : (n, 6) array
        array where rows are stress or strain vectors containing components of the symmetric Cauchy tensor in the order
        X,Y,Z,XY,YZ,XZ
    engineering_strain : bool, default = False
        if ``True`` the shear components in ``str_vectors`` are interpreted as being the engineering shear strains,
        which are twice the tensor shear strains, and a scaling factor of ``0.5`` is applied.

    Returns
    -------
    components : (n, 3) array
        principal component values. Each entry is ordered such that ``P1>P2>P3``.
    directions : (n, 3, 3) array
        principal component direction unit vectors, such that the entry columns ``directions[:,:,i]`` are the direction
        vectors corresponding to the component entries ``components[:,i]``.

    Examples
    --------
    >>> import numpy as np
    >>> from pytwin import principal_stress_strain
    >>> stress_vectors = np.array([[-10,50,0,40,0,0], [15,40,0,30,0,0]])
    >>> component, direction = principal_stress_strain(stress_vectors)
    >>> component
    array([[ 70.,   0., -30.],
           [ 60.,   0.,  -5.]])
    >>> direction
    array([[[-0.4472136 ,  0.        , -0.89442719],
            [-0.89442719,  0.        ,  0.4472136 ],
            [ 0.        ,  1.        ,  0.        ]],
           [[-0.5547002 ,  0.        , -0.83205029],
            [-0.83205029,  0.        ,  0.5547002 ],
            [ 0.        ,  1.        ,  0.        ]]])

    Calculate principal strain where engineering shear strains are used.
    >>> strain_vectors = np.array([[-2.1e-4, 0.0, 5.0e-5, 0.0, 0.0, 1.5e-6]])
    >>> component, direction = principal_stress_strain(strain_vectors, engineering_strain=True)
    >>> component
    array([[ 5.00086536e-05,  0.00000000e+00, -2.10008654e-04]])
    >>> direction
    array([[[-0.00576894,  0.        , -0.99998336],
            [ 0.        ,  1.        ,  0.        ],
            [-0.99998336,  0.        ,  0.00576894]]])
    """  # noqa: E501
    # Set multiplier to convert engineering shear strain to tensor shear strain.
    shear_scaling = 0.5 if engineering_strain else 1.0

    # Build tensor array by stacking vector of components
    str_tensors = _tensor_from_vector(str_vectors, shear_scaling)

    # Calculate principle components and directions
    principals, directions = np.linalg.eig(str_tensors)

    # Sort by decreasing eigenvalue size, since Numpy does not necessarily order eigenvalues.
    sort_order = np.flip(np.argsort(principals, axis=1), axis=1)
    principals = np.take_along_axis(principals, sort_order, axis=1)
    vec_sort_order = np.expand_dims(sort_order, axis=2)
    # Numpy returns eigenvectors such that each column corresponds to an eigenvalue. Transpose rows and columns to use
    # sort indexing, then return to original orientation.
    directions = np.transpose(directions, (0, 2, 1))
    directions = np.take_along_axis(directions, vec_sort_order, axis=1)
    directions = np.transpose(directions, (0, 2, 1))
    return principals, directions


def _tensor_from_vector(vectors: np.ndarray, shear_scaling):
    """
    Convert (n,6) vector array to (n,3,3) tensor array.

    Input vectors have the form ``[X, Y, Z, XY, YZ, XZ]``.

    Output tensors have the form ``[[X, XY, XZ], [XY, Y, YZ], [XZ, YZ, Z]]``.
    """
    # Build tensor array by stacking vector of components
    X = vectors[:, TENSOR_MAP["X"]]
    Y = vectors[:, TENSOR_MAP["Y"]]
    Z = vectors[:, TENSOR_MAP["Z"]]
    XY = vectors[:, TENSOR_MAP["XY"]] * shear_scaling
    YZ = vectors[:, TENSOR_MAP["YZ"]] * shear_scaling
    XZ = vectors[:, TENSOR_MAP["XZ"]] * shear_scaling
    return np.stack([X, XY, XZ, XY, Y, YZ, XZ, YZ, Z], axis=1).reshape((-1, 3, 3))
