Property calculation module
========================================

The property calculation module features three functions designed to calculate the permeance necessary for membrane process simulation. :py:mod:`CalculPermeance` function computes permeance in units of [mol/mm:sup:`2` bar], taking partial pressure, self-diffusivity, gas uptake, density, and membrane thickness as inputs. These input variables can be derived from experiments and computational simulations. The :py:mod:`CropMSD` function extracts molecular dynamics (MD) simulation results, presenting the mean-squared displacement over time in matrix form. Based on the MSD matrix processed through the :py:mod:`CropMSD` function, the :py:mod:`CalculSefDiff` function derives the self-diffusivity of the gas components using Einstein's relational theory. These functions empower users to effortlessly determine membrane properties without necessitating prior background knowledge. Refer to the method section for the theoretical basis employed in the module and the Supplementary Information for detailed usage of each function.


First, import the module into Python after installation.

.. code-block:: python

   from pySembrane.calculator import *

In this module, three main functions exist.

   1. Calculate membrane property (:py:mod:`CalculPermeance`) 
   2. Extract mean-squared displacement (:py:mod:`CropMSD`)
   3. Determine self diffuvisity from MSD   (:py:mod:`CalculSefDiff`)

Detailed description of each function are described in next senction. The explanation include function usage, related theroy, and function structure.

---------------------------------------------------------------

Usage
-------

1. Calculate membrane property
''''''''''''''''''''''''''''''''''''''''''''''


.. code-block:: python

   # Package import
   from pySembrane.calculator import *

   # Membrane property
   P = 6                         # Gas pressure (bar)
   q = 6.4                       # Gas uptake (mol/kg)
   rho = 1320e-9                 # Material density (kg/mm^3)

   # Fiber sizing
   D_inner = 100*1e-1            # Membrane inner diameter (mm)
   D_outer = 250*1e-1            # Membrane outer diameter (mm)
   thickness = (D_outer-D_inner)/2

   # Calculate permeance
   a_mem = CalculPermeance(P, D_mem*1e6, q, rho, thickness)
   print('Permeance (mol/(mm2 bar s)): ', a_mem)


2. Extract mean-squared displacement
''''''''''''''''''''''''''''''''''''''''''''''


.. code-block:: python

    # data import
    msd_crop = CropMSD('msd_self_methane_0.dat')


3. Determine self diffuvisity from MSD
''''''''''''''''''''''''''''''''''''''''''''''

.. code-block:: python

    # Calculate self-diffusivity using MSD
    D_mem = CalculSelfDiff(msd_crop)

--------------------------------------------------------------------------------


Function structures
------------------------------------------------

.. currentmodule:: pySembrane.calculator

.. autofunction:: CalculPermeance

.. currentmodule:: pySembrane.calculator

.. autofunction:: CropMSD

.. currentmodule:: pySembrane.calculator

.. autofunction:: CalculSelfDiff

------------------------------------------------


Theory
------------------------------------------------

Permeance estimation from self-diffusivity
''''''''''''''''''''''''''''''''''''''''''''''''''''

Among membrane properties, permeance is a crucial parameter, which signifies the efficiency of gas transport through a membrane. It is contingent on the thickness of the membrane and holds paramount importance in the simulation of membrane processes. The determination of permeance typically emanates from either empirical experimentation or computational simulations. However, due to time and cost efficiency, computational simulations are widely preferred in practice. Permeance is calculated from the following equation:

.. math::

    a_{i} = \frac{\mathcal{P}}{d} = \frac{q_{i}\mathcal{D}_{i}}{p_{i}\rho d}


where :math:`P`, :math:`q`, :math:`\mathcal{D}`, :math:`p`, and :math:`\rho` denote permeability, gas uptake, self-diffusion coefficient, partial pressure, and density of membrane, respectively. The subscript :math:`i` denotes each gas component. Gas uptake can be calculated through grand canonical Monte Carlo (MC) simulation, and self-diffusivity can be obtained from molecular dynamics (MD). 


Self-diffusivity estimation from mean-squared displacement
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

.. _MDsimDiffusivity:
.. figure:: images/MDsimDiffusivity.png
  :width: 700
  :align: center

  **Fig. 1** Schematics of self-diffusion coefficient calculation from molecular dynamics simulation.


Molecular dynamics is a computational method calculating the movements of atoms and molecules, then the evolving positions of individual molecules over time, allowing the derivation of gas molecule diffusion patterns. During this process, the molecule's position is represented as a vector in three-dimensional space (x, y, z axes). The MD simulation captures all molecules' positions through mean-squared displacement (MSDs), calculated using the Einstein formula. MSDs, rooted in Brownian motion studies, signify the average movement speed of particles. Over time, **Fig. 1** demonstrates the gradual dispersion of molecules, showing an increase in average distance. Self-diffusivities (:math:`\mathcal{D}`) with the desired dimensionality can be computed by fitting MSDs over time to a linear model. 


.. math::

    \lim_{t \to \infty}\left< \lVert r_{i}(t)-r_{i}(0) \rVert ^{2} \right> = 6 \mathcal{D}t


