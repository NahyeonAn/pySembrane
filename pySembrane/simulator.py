"""
====================================
 :mod:`simulator` module
====================================
This module define the hollow fiber membrane module and perform and analyze process simulation.
"""

#%% Library import
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# import Membrane_pack
from scipy.interpolate import interp1d

from scipy.optimize import minimize
from scipy import interpolate

parameters = {'axes.labelsize': 17,
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
          'axes.titlesize': 20}
plt.rcParams.update(parameters)

import pickle
from itertools import product
import warnings
import math

# Constants
Rgas = 8.314*1e9                # mm3 Pa/K mol

def SolveFDM(dy_fun, y0, t, args= None):
#    if np.isscalar(t):
#        t_domain = np.linspace(0,t, 10001, dtype=np.float64)
#    else:
#        t_domain = np.array(t[:], dtype = np.float64)
    t_domain = np.array(t[:], dtype = np.float64)
    y_res = []
    dt_arr = t_domain[1:] - t_domain[:-1]

    N = len(y0)
    tt_prev = t_domain[0]
    y_tmp = np.array(y0, dtype = np.float64)
    y_res.append(y_tmp)
    if args == None:
        for tt, dtt in zip(t_domain[:-1], dt_arr):
            dy_tmp = np.array(dy_fun(y_tmp, tt))
            y_tmp_new = y_tmp + dy_tmp*dtt
            tt_prev = tt
            y_res.append(y_tmp_new)
            y_tmp = y_tmp_new
#            if tt%10 == 1:
#                print(y_tmp_new, y_tmp)
        y_res_arr = np.array(y_res, dtype = np.float64)
    else:
        for tt, dtt in zip(t_domain[1:], dt_arr):
            dy_tmp = np.array(dy_fun(y_tmp, tt))
            y_tmp_new = y_tmp + dy_tmp*dtt
            tt_prev = tt
            y_res.append(y_tmp_new)
            y_tmp = y_tmp_new
        y_res_arr = np.array(y_res, dtype=object)
    
    return y_res_arr

#%%
class MembraneProc:
    def __init__(self, configuration, length, d_module, n_fiber, 
                 n_component, n_node = 10, sweep_gas = False):
        """Define hollow fiber membrane module

        Args:
            configuration (str): Membrane module configureation [COFS, COFT, CTFS, CTFT]
            length (float): Module length `(mm)`
            d_module (float): Module diameter `(mm)`
            n_fiber (integer): The number of fibers
            n_component (integer): The number of gas components
            n_node (int, optional): The number of nodes. Defaults to 10.
        """
        self._length = length
        self._d_module = d_module
        self._n_fiber = n_fiber
        
        self._n_comp = n_component
        self._n_node = int(n_node)
        self._sweep_gas = sweep_gas
        
        self._z = np.linspace(0, self._length, self._n_node+1)
        
        self._required = {'Design':False,
                        'Membrane_info':False,
                        'Gas_prop_info': False,
                        'Mass_trans_info': False,
                        'BoundaryC_info': False,
                        'InitialC_info': False}
        
        if len(configuration) != 4:
            print("Configuration should be FOUR capital alphbets: COFS, CTFS, COFT, CTFT")
            print("CO: co-current // CT: counter-current")
            print("FS: feed-shell side // FT: feed-tube side")
        else:
            self._config = configuration
            self._required['Design'] = True
        
    def __str__(self):
        str_return = '[[Current information included here]] \n'
        for kk in self._required.keys():
            str_return = str_return + '{0:16s}'.format(kk)
            if type(self._required[kk]) == type('  '):
                str_return = str_return+ ': ' + self._required[kk] + '\n'
            elif self._required[kk]:
                str_return = str_return + ': True\n'
            else:
                str_return = str_return + ': False\n'
        return str_return
    
    def membrane_info(self, a_perm, d_inner, d_outer):
        """Define membrane material property

        Args:
            a_perm (nd_array): Gas permeance for each component `(mol/(mm2 bar s))`
            d_inner (float): Fiber inner diameter `(mm)`
            d_outer (float): Fiber outer diameter`(mm)`
        """
        self._d_inner = d_inner
        self._d_outer = d_outer
        
        self._ac_shell = np.pi*self._d_module**2/4 - self._n_fiber*np.pi*self._d_inner**2/4     # (mm^2) 
        self._thickness = (self._d_outer-self._d_inner)/2
        
        if len(a_perm) != self._n_comp:
            print('Output should be a list/narray including {} narray!'.format(self._n_comp))
        else:
            self._a_perm = a_perm
            self._required['Membrane_info'] = True
        
    def gas_prop_info(self, molar_mass, mu_viscosity, rho_density,):
        """Define gas property

        Args:
            molar_mass (nd_array): Molar mass `(mol/kg)`
            mu_viscosity (nd_array): Visocosity `(Pa s)`
            rho_density (nd_array): Density `(kg/mm3)`
        """
        stack_true = 0
        if len(molar_mass) == self._n_comp:
            stack_true = stack_true + 1
        else:
            print('The input variable should be a list/narray with shape ({0:d}, ).'.format(self._n_comp))
            
        if len(mu_viscosity) == self._n_comp:
            stack_true = stack_true + 1
        else:
            print('The input variable should be a list/narray with shape ({0:d}, ).'.format(self._n_comp))
            
        if len(rho_density) == self._n_comp:
            stack_true = stack_true + 1
        else:
            print('The input variable should be a list/narray with shape ({0:d}, ).'.format(self._n_comp))    
            
        if stack_true == 3:
            self._molar_mass = molar_mass
            self._mu = mu_viscosity
            self._rho = rho_density
            self._required['Gas_prop_info'] = True
        
    def mass_trans_info(self, k_mass_transfer):
        """Define mass transfer information

        Args:
            k_mass_transfer (float): Mass transfer coefficient `(mm/s)`
        """
        self._k_mtc = k_mass_transfer
        self._required['Mass_trans_info'] = True
        
    def boundaryC_info(self,y_inlet, p_f_inlet, f_f_inlet, T_inlet, f_sweep = False):
        """ Determin boundary condition

        Args:
            y_inlet (nd_array): Gas composition in feed flow with shape (n_component, ).
            p_f_inlet (scalar): Feed pressure `(bar)`
            f_f_inlet (scalar): Feed flowrate `(mol/s)`
            T_inlet (scalar): Feed temperature `(K)`
            f_sweep (list or nd_array): Sweep gas flowarte of each component `(mol/s)`
        """
        try:
            if len(y_inlet) == self._n_comp:
                self._y_in = y_inlet
                self._Pf_in = p_f_inlet
                self._T_in = T_inlet
                self._Ff_in = f_f_inlet*y_inlet
                if self._sweep_gas:
                    if len(y_inlet) == self._n_comp:
                        self._f_sw = f_sweep
                    else:
                        print('The sweep gas flowrate should be a list/narray with shape (n_component, ).')
                else:
                    self._f_sw = np.zeros(self._n_comp)
                self._required['BoundaryC_info'] = True
            else:
                print('The inlet composition should be a list/narray with shape (n_component, ).')            
        except:
            print('The inlet composition should be a list/narray with shape (n_component, ).')
        
    # ODE
    def _COFS(self, y, z):
        F_f = np.array([y[ii] for ii in range(self._n_comp)])
        F_p = np.array([y[ii+self._n_comp] for ii in range(self._n_comp)])
        Pf, Pp =  y[self._n_comp*2], y[self._n_comp*2+1]

        F_f_tot = np.sum(F_f, axis=0)           # Minimum criteria?
        F_p_tot = np.sum(F_p, axis=0)

        x_i = F_f/F_f_tot
        y_i = F_p/F_p_tot
        
        mu_f = np.sum(self._mu*x_i)     # feed side viscosity (Pa s)
        mu_p = np.sum(self._mu*y_i)     # permeate side viscosity (Pa s)

        if self._cp:
            x_mem = self._CalculCP(Pf, Pp, x_i, y_i)
            Pf_i = x_mem*Pf
        else:
            Pf_i = x_i*Pf
        
        Pp_i = y_i*Pp
        
        dPfdz_den = - 192*self._n_fiber*self._d_outer*(self._d_module + self._n_fiber*self._d_outer)*Rgas*self._T_in*mu_f*F_f_tot
        dPfdz_num = np.pi*(self._d_module**2 - self._n_fiber*self._d_outer**2)**3 * Pf
        dPfdz = dPfdz_den/dPfdz_num*1E-10
        
        dPpdz_den = 128*Rgas*self._T_in*mu_p*F_p_tot
        dPpdz_num = np.pi*self._d_inner**4 * self._n_fiber * Pp
        dPpdz = dPpdz_den/dPpdz_num*1E-10
            
        J = self._a_perm*(Pf_i - Pp_i)
        arg_neg_J = J < 0
        J[arg_neg_J] = 0
        
        dF_f = -self._d_outer*np.pi*J*self._n_fiber
        dF_p = -self._d_outer*np.pi*J*self._n_fiber

        dF_p = -dF_p
        dPpdz = -dPpdz

        dF_f = dF_f.tolist()
        dF_p = dF_p.tolist()

        dydz = dF_f+ dF_p+ [dPfdz]+[dPpdz]
        return dydz
    
    def _COFT(self, y, z):
        F_f = np.array([y[ii] for ii in range(self._n_comp)])
        F_p = np.array([y[ii+self._n_comp] for ii in range(self._n_comp)])
        Pf, Pp =  y[self._n_comp*2], y[self._n_comp*2+1]

        F_f_tot = np.sum(F_f, axis=0)           # Minimum criteria?
        F_p_tot = np.sum(F_p, axis=0)

        x_i = F_f/F_f_tot
        y_i = F_p/F_p_tot
        
        mu_f = np.sum(self._mu*x_i)     # feed side viscosity (Pa s)
        mu_p = np.sum(self._mu*y_i)     # permeate side viscosity (Pa s)

        if self._cp:
            x_mem = self._CalculCP(Pf, Pp, x_i, y_i)
            Pf_i = x_mem*Pf
        else:
            Pf_i = x_i*Pf
        
        Pp_i = y_i*Pp
        
        dPfdz_den = - 128*Rgas*self._T_in*mu_f*F_f_tot     # note: mu_f vs. mu_p
        dPfdz_num = np.pi*self._d_inner**4 * self._n_fiber * Pf
        dPfdz = dPfdz_den/dPfdz_num*1E-10
        
        dPpdz_den = 192*self._n_fiber*self._d_outer*(self._d_module + self._n_fiber*self._d_outer)*Rgas*self._T_in*mu_p*F_p_tot
        dPpdz_num = np.pi*(self._d_module**2 - self._n_fiber*self._d_outer**2)**3 * Pp
        dPpdz = dPpdz_den/dPpdz_num*1E-10
            
        J = self._a_perm*(Pf_i - Pp_i)
        arg_neg_J = J < 0
        J[arg_neg_J] = 0
        
        dF_f = -self._d_outer*np.pi*J*self._n_fiber
        dF_p = -self._d_outer*np.pi*J*self._n_fiber

        dF_p = -dF_p
        dPpdz = -dPpdz

        dF_f = dF_f.tolist()
        dF_p = dF_p.tolist()

        dydz = dF_f+ dF_p+ [dPfdz]+[dPpdz]
        return dydz
    
    def _CTFS(self, y, z):
        F_f = np.array([y[ii] for ii in range(self._n_comp)])
        F_p = np.array([y[ii+self._n_comp] for ii in range(self._n_comp)])
        Pf, Pp =  y[self._n_comp*2], y[self._n_comp*2+1]

        F_f_tot = np.sum(F_f, axis=0)           # Minimum criteria?
        F_p_tot = np.sum(F_p, axis=0)

        x_i = F_f/F_f_tot
        y_i = F_p/F_p_tot
        
        mu_f = np.sum(self._mu*x_i)     # feed side viscosity (Pa s)
        mu_p = np.sum(self._mu*y_i)     # permeate side viscosity (Pa s)
        
        if self._cp:
            x_mem = self._CalculCP(Pf, Pp, x_i, y_i)
            Pf_i = x_mem*Pf
        else:
            Pf_i = x_i*Pf

        Pp_i = y_i*Pp
        
        dPfdz_den = - 192*self._n_fiber*self._d_outer*(self._d_module + self._n_fiber*self._d_outer)*Rgas*self._T_in*mu_f*F_f_tot
        dPfdz_num = np.pi*(self._d_module**2 - self._n_fiber*self._d_outer**2)**3 * Pf
        dPfdz = dPfdz_den/dPfdz_num*1E-10
        
        dPpdz_den = 128*Rgas*self._T_in*mu_p*F_p_tot
        dPpdz_num = np.pi*self._d_inner**4 * self._n_fiber * Pp
        dPpdz = dPpdz_den/dPpdz_num*1E-10
            
        J = self._a_perm*(Pf_i - Pp_i)
        arg_neg_J = J < 0
        J[arg_neg_J] = 0
        
        dF_f = -self._d_outer*np.pi*J*self._n_fiber
        dF_p = -self._d_outer*np.pi*J*self._n_fiber
        
        dF_f = dF_f.tolist()
        dF_p = dF_p.tolist()

        dydz = dF_f+ dF_p+ [dPfdz]+[dPpdz]
        return dydz
    
    def _CTFT(self, y, z):
        F_f = np.array([y[ii] for ii in range(self._n_comp)])
        F_p = np.array([y[ii+self._n_comp] for ii in range(self._n_comp)])
        Pf, Pp =  y[self._n_comp*2], y[self._n_comp*2+1]

        F_f_tot = np.sum(F_f, axis=0)           # Minimum criteria?
        F_p_tot = np.sum(F_p, axis=0)

        x_i = F_f/F_f_tot
        y_i = F_p/F_p_tot
        
        mu_f = np.sum(self._mu*x_i)     # feed side viscosity (Pa s)
        mu_p = np.sum(self._mu*y_i)     # permeate side viscosity (Pa s)
        
        if self._cp:
            x_mem = self._CalculCP(Pf, Pp, x_i, y_i)
            Pf_i = x_mem*Pf
        else:
            Pf_i = x_i*Pf
        
        Pp_i = y_i*Pp
        
        dPfdz_den = - 128*Rgas*self._T_in*mu_f*F_f_tot     # note: mu_f vs. mu_p
        dPfdz_num = np.pi*self._d_inner**4 * self._n_fiber * Pf
        dPfdz = dPfdz_den/dPfdz_num*1E-10
        
        dPpdz_den = 192*self._n_fiber*self._d_outer*(self._d_module + self._n_fiber*self._d_outer)*Rgas*self._T_in*mu_p*F_p_tot
        dPpdz_num = np.pi*(self._d_module**2 - self._n_fiber*self._d_outer**2)**3 * Pp
        dPpdz = dPpdz_den/dPpdz_num*1E-10
            
        J = self._a_perm*(Pf_i - Pp_i)
        arg_neg_J = J < 0
        J[arg_neg_J] = 0
        
        dF_f = -self._d_outer*np.pi*J*self._n_fiber
        dF_p = -self._d_outer*np.pi*J*self._n_fiber
        
        dF_f = dF_f.tolist()
        dF_p = dF_p.tolist()

        dydz = dF_f+ dF_p+ [dPfdz]+[dPpdz]
        return dydz
            
    def _find_Pp_in(self, y):
        Pp_in_list = np.linspace(1.000000001, 2, 50)
        Pp_out_list = []
        Pp_out_list = []
        
        for Pp_in in Pp_in_list:
            y[-1] = Pp_in
            y_res = SolveFDM(self._mem_model, y, self._z,)
            Pp_out = y_res[-1,5]
            Pp_out_list.append(Pp_out)

        P_reduced = np.array(Pp_out_list) - 1
        func = interpolate.UnivariateSpline(Pp_in_list, P_reduced, s=0)
        Pp_sol_list = func.roots()
        
        err_list = []
        if len(Pp_sol_list) > 1:
            for Pp_new in Pp_sol_list:
                y[-1] = Pp_new
                y_res = SolveFDM(self._mem_model, y, self._z,)
                
                Pp_out = y_res[-1,5]
                err_list.append((Pp_out-1)**2)
            Pp_sol = Pp_sol_list[np.argmin(np.array(err_list))]
            
        elif len(Pp_sol_list)==1:
            Pp_sol = Pp_sol_list[0]
        else:
            Pp_sol = Pp_in_list[0]
        return Pp_sol
   
    def initialC_info(self, on=True):
        """Derive (for co-current) or set (for counter-current) initial condition
        """
        
        if self._config[:2] == 'CO':
            self._Pp_in = 1.01
            
        elif self._config[:2] == 'CT':
            # Fp_init = np.array([self._Ff_in[ii]*0.5 for ii in range(self._n_comp)])
            self._Pp_in = 1
            
        if on:
            Pf_i = self._Pf_in*self._y_in
            Pp_i = self._Pp_in*self._y_in
                
            W_int = np.pi*self._d_outer*self._n_fiber*self._length
            Fp_curr, Fp_prev = np.zeros(self._n_comp),np.zeros(self._n_comp)
            
            if self._sweep_gas:
                if self._config[:2] == 'CO':
                    Fp_init = np.array(self._f_sw)
                else:
                    Fp_init = 0.05*self._Ff_in + self._f_sw
            
            else:
                if self._config[:2] == 'CO':
                    W_int = W_int/self._n_node
                    
                for ii in range(20000):
                    Fp_prev = Fp_curr
                    Fp_curr = self._a_perm*W_int*(Pf_i-Pp_i)/np.log(Pf_i/Pp_i)
                    Fp_tot = sum(Fp_curr)
                    
                    yi = Fp_curr/Fp_tot
                    Pp_i = yi*self._Pp_in          # value?

                    err = sum(np.abs(Fp_curr-Fp_prev))
                    if err < 1e-7:
                        break
            Fp_init = Fp_curr
            
        else:
            if self._sweep_gas:
                if self._config[:2] == 'CO':
                    Fp_init = np.array(self._f_sw)
                else:
                    Fp_init = 0.05*self._Ff_in + self._f_sw
            else:
                if self._config[:2] == 'CO':
                    Fp_init = np.array([1e-6]*self._n_comp)
                else:
                    Fp_init = 0.05*self._Ff_in

        y0 = np.array(list(self._Ff_in) + list(Fp_init) + [self._Pf_in, self._Pp_in])       
        self._y0 = y0
        self._required['InitialC_info'] = True

    def _CalculCP(self, Pf, Pp, x_f, y_p):
        A_vol = self._a_perm*self._molar_mass/self._rho      # Volumetric permeance
        k = self._k_mtc
        n_comp = self._n_comp
        P_ref, T_ref = self._cp_cond
        X = self._Pf_in/P_ref*T_ref/self._T_in
        M_mtr = np.zeros((n_comp-1, n_comp-1))
        for jj in range(n_comp-1):
            for ii in range(n_comp-1):
                if ii == jj:
                    M_mtr[ii, jj] = A_vol[ii]*Pf+k*X-x_f[ii]*Pf*(A_vol[ii]-A_vol[-1])
                else:
                    M_mtr[ii, jj] = -x_f[jj]*Pf*(A_vol[ii]-A_vol[-1])
        
        M_inv = np.linalg.inv(M_mtr)
        sum_y = np.sum([A_vol[ii]*y_p[ii] for ii in range(n_comp-1)])
        Y = A_vol[-1]*Pf-A_vol[-1]*Pp*y_p[-1]-Pp*sum_y
        b = [A_vol[ii]*Pp*y_p[ii]+k*X*x_f[ii]+x_f[ii]*Y for ii in range(n_comp-1)]

        x_n_1 = np.dot(M_inv,b)
        x_mem = np.insert(x_n_1, n_comp-1, 1-sum(x_n_1))
        return x_mem
    
    def run_mem(self, tolerance=1e-7, iteration=20000, Kg=0.1, cp=False, cp_cond = False):
        """Run membrane process simulation

        Args:
            tolerance (float, optional): Tolerance. Defaults to 1e-7.
            iteration (int, optional): Iteration. Defaults to 20000.
        """
        print('Simulation started')
        self._cp = cp       # Concentration polarization
        self._cp_cond = cp_cond     # list [P_ref, T_ref]
        
        if self._config == 'COFS':
            model = self._COFS
        elif self._config == 'COFT':
            model =  self._COFT
        elif self._config == 'CTFS':
            model =  self._CTFS
        elif self._config == 'CTFT':
            model =  self._CTFT

        for ii in range(iteration):
            y_res = SolveFDM(model, self._y0, self._z,)
            F_f = np.array([y_res[:,ii] for ii in range(self._n_comp)])
            F_p = np.array([y_res[:,ii+self._n_comp] for ii in range(self._n_comp)])
            fp_0_i = np.array([y_res[0,ii+self._n_comp] for ii in range(self._n_comp)])
            Pf, Pp = y_res[:,self._n_comp*2],y_res[:,self._n_comp*2+1]
            
            x_i = F_f/np.sum(F_f, axis=0)
            y_i = F_p/np.sum(F_p, axis=0)

            if self._cp:
                x_mem = np.array([self._CalculCP(Pf[ii], Pp[ii], x_i[:,ii], y_i[:,ii]) for ii in range(len(Pf))]).T
                Pf_i = x_mem*Pf
            else:
                Pf_i = x_i*Pf
            Pp_i = y_i*Pp
            
            J = (self._a_perm).reshape(-1,1)*(Pf_i - Pp_i)#*1E5
            arg_neg_J = J < 0
            J[arg_neg_J] = 0

            #Error calculation  
            _factor = np.pi*self._d_outer*self._length/self._n_node*self._n_fiber 
            
            # if self._config[:2] == 'CT':
            #     err = [(ffp-_factor*sum(J[ii,:])-fsw)/ffp for ii, (ffp, fsw)in enumerate(zip(fp_0_i,self._f_sw))]
            
            if self._config[:2] == 'CT':
                err = [(ffp-_factor*sum(J[ii,:]))/ffp for ii, (ffp, fsw)in enumerate(zip(fp_0_i,self._f_sw))]
            else:
                if self._sweep_gas:
                    err = [0 for ii, ffp in enumerate(fp_0_i)]
                else:
                    err = [(ffp-_factor*(J[ii,0]))/ffp for ii, ffp in enumerate(fp_0_i)]
                err_pp = (1-Pp[-1])/Pp[-1]
                err.append(err_pp) 
            
            tol = sum([abs(_err) for _err in err])
        
            Kg = 0.1
            for jj, _err in enumerate(err):
                if jj < self._n_comp:
                    fp_0_i[jj] = fp_0_i[jj]-Kg*_err*fp_0_i[jj]
                else:
                    self._Pp_in = self._Pp_in + Kg*_err*Pp[-1]
            self._y0 = np.array(list(self._Ff_in) +list(fp_0_i)+ [self._Pf_in, self._Pp_in])                
            if abs(tol) < tolerance:
                break
            if ii == iteration-1:
                print('Warning: Covergence failed!')
                break            
        self._y_res = y_res
        self.iteration = ii
        self.NoticeResultsCondition()
        return self._y_res
    
    def NoticeResultsCondition(self):
        y = self._y_res
        neg_y = y<0
        if sum(sum(neg_y[:, :self._n_comp]))>0:
            print('Warning: Negative flowrate is detected in retentate side')
        elif sum(sum(neg_y[:, self._n_comp:self._n_comp*2]))>0:
            print('Warning: Negative flowrate is detected in permeate side')
        else:
            print('Simulation is completed without warning')
    
    def MassBalance(self,):
        """Calculate mass balance error

        Returns:
            float: Percentage error `(%)`
        """
        y = self._y_res
        if self._config[:2] == 'CO':
            inpt = sum(y[0,:self._n_comp*2])
            outp = sum(y[-1,:self._n_comp*2])
        elif self._config[:2] == 'CT':
            inpt = sum(y[0,:self._n_comp])+sum(y[-1,self._n_comp:self._n_comp*2])
            outp = sum(y[-1,:self._n_comp])+sum(y[0,self._n_comp:self._n_comp*2])
        
        err = abs(inpt-outp)/inpt*100
        print('Mass balance (error %): ', err)
        return err
    
    def _CalculCompr(self, h_capa_ratio, comp_eiff):
        P_ref = 1                    # inlet pressure (1bar)
        total_CR = self._Pf_in/1   # total compression ratio
        R_gas = 8.314                # Gas constant (J/K mol)
        NN = int(np.log(total_CR)/np.log(2.5))+1     # # of compressors
        cost = 0
        work = 0
        for i in range(NN):
            effi = comp_eiff-(i*0.05)
            if i != NN-1:
                work_tmp = np.sum(self._Ff_in)*R_gas*self._T_in/effi*(h_capa_ratio/(h_capa_ratio-1))*((2.5*P_ref/P_ref)**((h_capa_ratio-1)/h_capa_ratio)-1)
            else:
                work_tmp = np.sum(self._Ff_in)*R_gas*self._T_in/effi*(h_capa_ratio/(h_capa_ratio-1))*((self._Pf_in/P_ref)**((h_capa_ratio-1)/h_capa_ratio)-1)
            
            work += work_tmp
            cost_tmp = 5840 * (work_tmp * 0.001)**0.82
            cost += cost_tmp
            P_ref = 2.5*P_ref
        return work*0.001, cost  


    def CalculTAC(self, unit_m_cost, mem_life, yr, interest_rate, h_capa_ratio, comp_eiff):
        """Calcuate total annualized cost

        Args:
            unit_m_cost (float): Unit membrane cost `(USD/ft2)`
            mem_life (int): Membrane life `(years)`
            yr (int): Project year `(years)`
            interest_rate (float): Interest rate
            h_capa_ratio (float): Heat capacity ratio
            comp_eiff (float): Compressor efficiency

        Returns:
            float: Total annualized cost `(USD/yr)`
        """
        area = np.pi*self._d_outer*self._length*self._n_fiber     # membrane area (mm2)
        c_work, c_cost = self._CalculCompr(h_capa_ratio, comp_eiff,)
        mem_cost = unit_m_cost * area
        
        capex = c_cost+mem_cost # FC
        TCI = 1.4*capex     # sum up working capital, fixed capital etc
        
        AF = (1-(1/(1+interest_rate)**yr))/interest_rate      # Annualized factor
        EAC = TCI / AF              # Annualized capital cost
        # TFI = 1.344*(area*MODP+c_cost)
        
        # opex
        FC = capex*0.014
        MRC = (mem_cost/2)/mem_life        # membrane replacement cost
        elect_cost = c_work*0.071    # 전기요금
        M = capex * 0.01              # Maintenance
        
        TPC = (FC + MRC + M*1.6 + elect_cost)/(1-0.26)    # Total product cost
        TAC = EAC + TPC             # Total annalized cost
        return TAC
    
    ### Plot ####
    def PlotResults(self, z_ran=False, component = False):
        """Plot simulation results

        Args:
            z_ran (list, optional): z-axis domain [min, max]. Defaults to False.
            component (list, optional): The name of gas components. Defaults to False.
        """
        if component == False:
            component = ['{'+f'{i}'+'}' for i in range(1,self._n_comp+1)]

        c_list = ['b', 'r', 'k', 'green', 'orange']
        line_list = ['-', '--', '-.',':']
        y_plot = self._y_res
        f_f = np.array([y_plot[:,i] for i in range(self._n_comp)])
        f_p = np.array([y_plot[:,i+self._n_comp] for i in range(self._n_comp)])
        Pf, Pp = y_plot[:,self._n_comp*2], y_plot[:,self._n_comp*2+1]
        x_i = f_f/np.sum(f_f, axis=0)
        y_i = f_p/np.sum(f_p, axis=0)
        
        if self._cp:
                x_mem = np.array([self._CalculCP(Pf[ii], Pp[ii], x_i[:,ii], y_i[:,ii]) for ii in range(len(Pf))]).T
                Pf_i = x_mem*Pf
        else:
            Pf_i = x_i*Pf
        Pp_i = y_i*Pp
            
        J =(Pf_i - Pp_i) * self._a_perm.reshape(-1, 1)
        arg_neg_J = J < 0
        J[arg_neg_J] = 0
        
        if self._config[:2] == 'CO':
            dPp = (Pp[0]-Pp)*1e5
        elif self._config[:2] == 'CT':
            dPp = (Pp[-1]-Pp)*1e5

        ########### flux  ##########
        fig = plt.figure(figsize=(10,7),dpi=90)
        fig.subplots_adjust(hspace=0.5, wspace=0.3)
        ax1 = fig.add_subplot(221)
        for i in range(self._n_comp):
            ax1.plot((self._z*1e-3), (J[i]*1e6), linewidth=2,color = c_list[0], 
                        linestyle= line_list[i], label=f'J$_{component[i]}$')
        ax1.set_xlabel('z (m)')
        ax1.set_ylabel('fluxes [mol/(m2 s)]')
        ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax1.legend(fontsize=13, loc='best')
        # plt.xlim([0, z_dom[-1]*1e-3])
        if z_ran:
            plt.xlim(z_ran)
        ax1.grid(linestyle='--')
        
        ########### Flowrate  ##########
        ax2 = fig.add_subplot(222)
        for i in range(self._n_comp):
            ax2.plot(self._z*1e-3, f_f[i], linewidth=2,color = c_list[0],
                        linestyle= line_list[i], label=f'Feed$_{component[i]}$')
        ax2.set_xlabel('z (m)')
        ax2.set_ylabel('feed flowrate (mol/s)')
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax2.grid(linestyle='--')
        ax3 = ax2.twinx()
        for i in range(self._n_comp):
            ax3.plot(self._z*1e-3, f_p[i], linewidth=2,color = c_list[1],
                        linestyle= line_list[i], label=f'Perm$_{component[i]}$')
        ax3.set_ylabel('Permeate flowrate (mol/s)')
        ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax2.yaxis.label.set_color(c_list[0])
        ax3.yaxis.label.set_color(c_list[1])
        ax3.spines["right"].set_edgecolor(c_list[1])
        ax3.spines["left"].set_edgecolor(c_list[0])
        if z_ran:
            plt.xlim(z_ran)
        ax2.tick_params(axis='y', colors=c_list[0])
        ax3.tick_params(axis='y', colors=c_list[1])
        # plt.xlim([0,0.1])

        ########### Mole fraction ##########
        ax4 = fig.add_subplot(223)
        for i in range(self._n_comp):
            ax4.plot((self._z*1e-3), x_i[i], linewidth=2, color=c_list[0],
                        linestyle=line_list[i], label=f'x$_{component[i]}$')
        ax4.set_xlabel('z (m)')
        plt.ylim([0, 1]) 
        ax4.set_ylabel('mole fraction (mol/mol)')
        ax4.grid(linestyle='--')
        ax5 = ax4.twinx()
        for i in range(self._n_comp):
            ax5.plot((self._z*1e-3), y_i[i], linewidth=2, color=c_list[1], 
                        linestyle=line_list[i], label=f'y$_{component[i]}$')
        plt.ylim([-0.01, 1.01])    
        if z_ran:
            plt.xlim(z_ran)
        ax4.yaxis.label.set_color(c_list[0])
        ax5.yaxis.label.set_color(c_list[1])
        ax4.tick_params(axis='y', colors=c_list[0])
        ax5.tick_params(axis='y', colors=c_list[1])
        ax5.spines["right"].set_edgecolor(c_list[1])
        ax5.spines["left"].set_edgecolor(c_list[0])
        
        ########### Pressure drop ##########
        ax6 = fig.add_subplot(224)
        ax6.plot(self._z*1e-3, (Pf[0]-Pf)*1e5, 'b-', label = 'Feed side')
        ax6.set_xlabel('z (m)')
        ax6.set_ylabel('$\\vartriangle$ $P_{f}$ (Pa)')
        ax6.ticklabel_format(axis='y', style='plain')
        ax6.grid(linestyle='--')
        ax7= ax6.twinx()
        ax7.plot(self._z*1e-3, dPp, 'r-', label = 'Permeate side')
        ax7.set_ylabel('$\\vartriangle$ $P_{p}$ (Pa)')
        fig.tight_layout()
        ax6.yaxis.label.set_color('b')
        ax7.yaxis.label.set_color('r')
        ax6.tick_params(axis='y', colors='b')
        ax7.tick_params(axis='y', colors='r')
        
        ax7.spines["right"].set_edgecolor('r')
        ax7.spines["left"].set_edgecolor('b')
        # plt.xlim([0, 0.005])
        if z_ran:
            plt.xlim(z_ran)
        plt.show()
            
                         
# #%%
# #%%
# # Sizing parameters
# D_inner = 200*1e-3            # Membrane inner diameter (mm)
# D_outer = 250*1e-3            # Membrane outer diameter (mm)
# D_module = 0.1*1e3            # Module diameter (mm)
# N_fiber = 60000               # number of fiber (-)
# L = 0.6*1e3                   # fiber length (mm)
# n_component = 2

# mem = Membrane(L, D_inner, D_outer, D_module, N_fiber, n_component, N_node = 1e3)
# print(mem)
# # %%
# a_perm = np.array([3.207e-9, 1.33e-10])*1e-6*1e5 #Permeance(mol/(mm2 bar s))
# mem.membrane_info(a_perm)
# print(mem)
# # %%
# Mw_i = np.array([44e-3, 16e-3])     # Molar weight (kg/mol)
# rho_i = np.array([1.98, 0.657])*1e-9     # Density (kg/mm3)
# mu_H2 = 0.0155e-3           # H2 viscosity (Pa s)
# mu_N2 = 0.011e-3           # N2 viscosity (Pa s)
# # viscosity values from https://www.engineeringtoolbox.com/gases-absolute-dynamic-viscosity-d_1888.html
# mu_i = np.array([mu_H2, mu_N2])

# mem.gas_prop_info(Mw_i, mu_i, rho_i)
# print(mem)
# # %%
# k_mass = 1e-1               # Mass transfer coeff. (mm/s)
# mem.mass_trans_info(k_mass)
# print(mem)
# # %%
# # Operating conditions
# P_feed = 60                # pressure of feed side (bar)
# T = 296.15
# y_feed = np.array([0.1, 0.9])     # mole fraction (CO2, CH4)
# f_f_inlet = 0.175
# Ff_z0_init = list(y_feed*f_f_inlet)

# mem.boundaryC_info(P_feed, T, y_feed, Ff_z0_init)
# print(mem)
# # %%
# Fp_H2_in = 1e-6     # initial value
# Fp_N2_in = 1e-6     # initial value
# Fp_init = np.array([Fp_H2_in,Fp_N2_in])
# Pp_z0 = 1.01      # initial guess

# mem.initialC_info(Fp_init, Pp_z0)
# print(mem)
# # %%
# res = mem.run_mem(2, 'P2F')
# # %%

# err = mem.MassBalance()
# # %%
# mem.results_plot_kci()
# # %%

# %%
