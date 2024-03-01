Examples
========================

The application of pySembrane is demonstrated through an example of membrane screening for green H\ :sub:`2` separation. Practical case studies performed with the package and discussion of the results illustrate how pySembrane can be beneficially applied across various research fields and emphasize the reliability and compatibility of the package.

.. figure:: images/CasestudiesScheme.png
  :width: 700
  :alt: CasestudiesScheme
  :align: center

  **Fig. 1** Target system and hierarchical workflow for evaluating metal-organic framework (MOF) membrane based on three sequential examples. In the first example (1), the membrane process simulation is employed to screen potential candidates from numerous MOF databases based on product purity. Subsequently, in the second example (2), process optimization is conducted to maximize economic feasibility for the chosen MOF candidate, leading to the identification of the most promising MOF. In the third example (3), an integrated analysis involving a commercial program (Aspen Plus) and pySembrane is executed, offering an in-depth analysis of the plant-scale performance of the selected MOF.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Example1
   Example2
   Example3
   

----

**Problem deifinition and process description**

For the optimal MOF selection of a green hydrogen production process, pySembrane is applied to three examples to show how to utilize pySembrane in actual problems, as shown in **Fig. 1**. In recent years, there has been significant interest in green hydrogen, owing to its potential as an environmentally friendly alternative to traditional fuels. Simultaneously, green ammonia (NH\ :sub:`3`) is gaining considerable attention as a transportation medium for hydrogen (H\ :sub:`2`). Given that NH\ :sub:`3` already possesses well-established infrastructure for production, transportation, and storage, green H\ :sub:`2` generated from regions abundant in renewable energy sources, such as Australia, Saudi Arabia, Chile, and Argentina, can be efficiently transported over long distances to global markets in the form of NH\ :sub:`3`, as illustrated in **Fig. 1** (a). Ensuring the high purity of H\ :sub:`2` post-decomposition of transported green NH\ :sub:`3` is crucial to meet the required purity standards for various applications. Among the various technologies for refining H\ :sub:`2`, the membrane-based gas separation process is in the spotlight for light gas separation owing to advantages such as high energy efficiency and ease of operation. However, due to significant performance variations in the membrane process depending on the material, selecting the optimal membrane requires thorough evaluation. Thus, across the three examples, membrane materials undergo evaluation through stages—single metric, optimization, and plant scale—to ultimately determine the optimal MOF membrane for green H\ :sub:`2` separation.


In the first example, potential MOFs were selected based on the process performance of the numerous MOFs. Molecular and molecular dynamic simulations were performed to derive the membrane properties of each MOF, and then MOFs with purity exceeding 99.9\% were selected through the membrane process simulation. In the second example, the potential MOF selected in the previous step was analyzed in greater depth via process optimization. Given that both product purity and economic viability vary with process variables, the optimal MOF was identified by determining the best process conditions through economic analysis and optimization. In the last example, the performance of the best MOF selected in the previous step was analyzed at the plant scale. The plant-scale model, developed using Aspen Plus commercial software, was integrated with the membrane process model to evaluate the impact of process conditions on both the membrane process and the final product.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Database and pre-screening**


To find the most promising MOF, CoRE MOF 2019 database, which contains 12,020 MOFs, was analyzed for membrane materials. Out of these MOFs, 450 were excluded because of the absence of structure files (.cif) required for subsequent molecular simulations. The database provides structural properties of MOFs such as accessible surface area (ASA\_m2\_cm3), void fraction (AV\_VF), the largest cavity diameter (LCD), pore limiting diameter (PLD), largest sphere along the free path (LFPD), and volume (cm3\_g), among others. The distribution and correlations of each property are depicted in **Fig. 2**.

.. figure:: images/CoREMOFvariables.png
  :width: 500
  :alt: CoREMOFvariables
  :align: center

  **Fig. 2** CoRE MOF database property distributions

Before evaluating MOFs as membrane materials through three examples, pre-screening is carried out for efficient assessment. Pre-screening consists of three stages: removal of inconsistent structures, elimination of MOFs containing precious metals, and structural-based screening. Initially, MOFs with disordered structures or containing precious metals were excluded. Disordered MOFs need to be removed since they represent physically implausible structures, despite being listed in the database. Secondly, MOFs containing precious metals were eliminated due to their impracticality in large-scale industrial applications, as their high costs render them unsuitable. **Table 1** lists excluded precious metals. Lastly, the pore limiting diameter (PLD) represents the smallest pore diameter of a material, and if the PLD is smaller than the gas molecules, these gas molecules cannot penetrate the pores. As PLD is the most fundamental physical characteristic of a membrane, initial screening should be based on PLD values. In three examples, since the objective is to permeate the N\ :sub:`2`/H\ :sub:`2` mixture through the membrane, the diameter of N\ :sub:`2` was selected as the threshold. Consequently, 6,086 MOFs with a PLD > 3.31 were screened for further analysis.



.. table:: **Table 1** List of 25 precious metals that were excluded from the simulation.
  :align: center

  +------------------+---------------+
  | Chemical Symbols |     Name      |
  +==================+===============+
  |        Au        |     Gold      |
  +------------------+---------------+
  |        Ag        |    Silver     |
  +------------------+---------------+
  |        Pt        |   Platinum    |
  +------------------+---------------+
  |        Pd        |  Palladium    |
  +------------------+---------------+
  |        Ru        |  Ruthenium    |
  +------------------+---------------+
  |        Rh        |   Rhodium     |
  +------------------+---------------+
  |        Y         |   Yttrium     |
  +------------------+---------------+
  |        In        |    Indium     |
  +------------------+---------------+
  |        Hf        |   Hafnium     |
  +------------------+---------------+
  |        Ga        |    Galliu     |
  +------------------+---------------+
  |        Dy        |  Dysprosium   |
  +------------------+---------------+
  |        Pr        | Praseodymium  |
  +------------------+---------------+
  |        Te        |  Tellurium    |
  +------------------+---------------+
  |        U         |   Uranium     |
  +------------------+---------------+
  |        Se        |   Selenium    |
  +------------------+---------------+
  |        Mo        |  Molybdenum   |
  +------------------+---------------+
  |        Sm        |   Samarium    |
  +------------------+---------------+
  |        Nd        |  Neodymium    |
  +------------------+---------------+
  |        Pr        | Praseodymium  |
  +------------------+---------------+
  |        Gd        |  Gadolinium   |
  +------------------+---------------+
  |        Ir        |   Iridium     |
  +------------------+---------------+
  |        Eu        |   Europium    |
  +------------------+---------------+
  |        La        |  Lanthanum    |
  +------------------+---------------+
  |        Tb        |   Terbium     |
  +------------------+---------------+
  |        Tm        |   Thulium     |
  +------------------+---------------+

-----------------------------------------------------------------------------------------------------