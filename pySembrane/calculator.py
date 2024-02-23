import os
import numpy as np
from sklearn.linear_model import LinearRegression

def is_float(string):
    """ True if given string is float else False"""
    try:
        return float(string)
    except ValueError:
        return False
    
def CropMSD(filename):
    """Read Mean squared displacement data

    Args:
        filename (string): MSD data file

    Returns:
        _type_: ndarray
    """
    msd = []
    time = []
    with open(filename, 'r') as f:
        d = f.readlines()
        for i in d:
            k = i.rstrip().split()
            if is_float(k[1]):
                msd.append(float(k[1])) 
                time.append(float(k[0]))
    
    msd = np.array([time, msd]).reshape(2, -1)
    return msd

def CalculSelfDiff(msd):
    """Calculate self diffusivity from Mean squared displacement using Einstein relation.

    Args:
        msd (nd array): Mean squared displacement (A^2) according to time (ps)

    Returns:
        _type_: float
    """
    reg = LinearRegression(fit_intercept = False).fit(msd[0].reshape(-1, 1), msd[1].reshape(-1, 1))
    diff = reg.coef_[0]*1e-20*1e12   #A2/ps ==> m2/s
    diff = diff[0]
    print("Self diffusiivty (m^2/s): ", diff)
    return diff

def CalculPermeance(P_i, D_i, q_i, rho_i, thickness):
    """Calculate gas permeance

    Args:
        P_i (float): Gas pressure (bar)
        D_i (float): Self diffusivity (mm2/s)
        q_i (float): Gas uptake (mol/kg ads)
        rho_i (float): Molecule density (kg ads/mm3)
        thickness (float): Membrane thickness (mm)

    Returns:
        _type_: float
    """
    # Permeability(mol mm2/(bar s mm3) )
    p_i =  q_i*D_i / P_i * rho_i 

    #Permeance(mol/(mm2 bar s)) = Permeability(mol mm2/(bar s mm3)) / mm
    a_i = p_i/thickness               

    return a_i