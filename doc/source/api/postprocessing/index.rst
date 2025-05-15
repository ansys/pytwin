.. _ref_index_postprocessing:

Post-processing functions
=========================

PyTwin functions to perform physics-specific post-processing operations.

.. currentmodule:: pytwin

.. autosummary::
   :toctree: _autosummary

   stress_strain_component

Workflow example
----------------

This code show how the functions may be used to calculate equivalent strain for a material with Poisson's ratio of 0.3
(e.g. steel) where input strains lie in the XZ plane. See :ref:`ref_example_TBROM_FEA_static_structural_optimization`
for further examples where the function is applied to :class:`TwinModel <pytwin.TwinModel>` results.

.. code-block:: pycon

    >>> import numpy as np
    >>> from pytwin import stress_strain_component
    >>> strain_vectors = np.array([[-2.1e-4, 0.0, 5.0e-5, 0.0, 0.0, 1.5e-6]])
    >>> E_vonMises = stress_strain_component(strain_vectors, "E", "EQV", effective_pr=0.3)
    >>> E_vonMises
    array([0.00018382])
