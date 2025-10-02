#%%
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# import Membrane_pack
from scipy.interpolate import interp1d

from scipy.optimize import minimize
from scipy import interpolate

parameters = {'font.size': 20,                    # 기본 폰트 크기
              'font.family': 'Arial',
              'xtick.direction': 'in', # x축 눈금을 안쪽으로
              'ytick.direction': 'in', # y축 눈금을 안쪽으로
              }
plt.rcParams.update(parameters)

import pickle
from itertools import product
import warnings
import math

#%%

# Constants
Rgas = 8.314*1e9                # mm3 Pa/K mol
# Rgas = 8.314*1e-5                # m3 bar/K mol

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
class PFM_simple:
    def __init__(self, config, channel_num, channel_size, n_component, n_node = 10, sweep_gas = False):
        """Define hollow fiber membrane module
        """
        
        self._config = config
        self.channel_num = channel_num
        self.channel_L = channel_size[0] 
        self.channel_w = channel_size[1] 
        self.channel_h = channel_size[2] 

        self._n_comp = n_component
        self._n_node = int(n_node)
        self._sweep_gas = sweep_gas
        
        self._z = np.linspace(0, self.channel_L, self._n_node+1)
        
        self._required = {'Design':False,
                        'Membrane_info':False,
                        'Gas_prop_info': False,
                        'Mass_trans_info': False,
                        'BoundaryC_info': False,
                        'InitialC_info': False}
    
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
    
    
    def membrane_info(self, a_perm, thickness):
        """Define membrane material property

        Args:
            a_perm (nd_array): Gas permeance for each component `(mol/(mm2 bar s))`
            d_inner (float): Fiber inner diameter `(mm)`
            d_outer (float): Fiber outer diameter`(mm)`
        """

        self.thickness = thickness
        
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
    
    def _channel_co(self, y, z):
        F_f = np.array([y[ii] for ii in range(self._n_comp)])       # mol/s
        F_p = np.array([y[ii+self._n_comp] for ii in range(self._n_comp)])
        Pf, Pp =  y[self._n_comp*2], y[self._n_comp*2+1]            # bar

        F_f_tot = np.sum(F_f, axis=0)           # Minimum criteria?
        F_p_tot = np.sum(F_p, axis=0)

        x_i = F_f/F_f_tot
        y_i = F_p/F_p_tot
        
        mu_f = np.sum(self._mu*x_i)     # feed side viscosity (Pa s) 
        mu_p = np.sum(self._mu*y_i)     # permeate side viscosity (Pa s)
        
        
        M_mix_f = np.sum(x_i * self._molar_mass)  # feed side 평균 분자량 (kg/mol)
        rho_f = (Pf *1e5* M_mix_f) / (Rgas * self._T_in)    # kg/mm3

        M_mix_p = np.sum(y_i * self._molar_mass)  # permeate side 평균 분자량 (kg/mol)
        rho_p = (Pp *1e5* M_mix_p) / (Rgas * self._T_in)

        if self._cp:
            x_mem = self._CalculCP(Pf, Pp, x_i, y_i)
            Pf_i = x_mem*Pf
        else:
            Pf_i = x_i*Pf
        
        Pp_i = y_i*Pp
        ## 수정필요
        
        dPfdz = -12*mu_f/(self.channel_h**3 * self.channel_w)*(F_f_tot/self.channel_num* Rgas * self._T_in)/(Pf*1e5)*1e-5
        # 모든 채널로 동일한 유량 흐름

        dPpdz = -12*mu_p/(self.channel_h**3 * self.channel_w)*(F_p_tot/self.channel_num* Rgas * self._T_in)/(Pp*1e5)*1e-5
        # 맨위, 맨 아래는 유량 다른데..
        
        ## 
        J = self._a_perm*(Pf_i - Pp_i) # mol/(mm2 bar s) * bar = mol/(mm2 s)
        arg_neg_J = J < 0
        J[arg_neg_J] = 0
        
        dF_f = -2*self.channel_num*self.channel_w*J     # mm* mol/(mm2 s) = mol/(mm s)
        dF_p = -2*self.channel_num*self.channel_w*J

        dF_p = -dF_p
        dPpdz = dPpdz

        dF_f = dF_f.tolist()
        dF_p = dF_p.tolist()

        dydz = dF_f+ dF_p+ [dPfdz]+[dPpdz]

        return dydz
    
    def _channel_ct(self, y, z):
        F_f = np.array([y[ii] for ii in range(self._n_comp)])       # mol/s
        F_p = np.array([y[ii+self._n_comp] for ii in range(self._n_comp)])
        Pf, Pp =  y[self._n_comp*2], y[self._n_comp*2+1]            # bar

        F_f_tot = np.sum(F_f, axis=0)           # Minimum criteria?
        F_p_tot = np.sum(F_p, axis=0)

        x_i = F_f/F_f_tot
        y_i = F_p/F_p_tot
        
        mu_f = np.sum(self._mu*x_i)     # feed side viscosity (Pa s) 
        mu_p = np.sum(self._mu*y_i)     # permeate side viscosity (Pa s) 
        
        
        M_mix_f = np.sum(x_i * self._molar_mass)  # feed side 평균 분자량 (kg/mol)
        rho_f = (Pf *1e5* M_mix_f) / (Rgas * self._T_in)    # kg/mm3

        M_mix_p = np.sum(y_i * self._molar_mass)  # permeate side 평균 분자량 (kg/mol)
        rho_p = (Pp *1e5* M_mix_p) / (Rgas * self._T_in)

        if self._cp:
            x_mem = self._CalculCP(Pf, Pp, x_i, y_i)
            Pf_i = x_mem*Pf
        else:
            Pf_i = x_i*Pf
        
        Pp_i = y_i*Pp

        # A_chan = self.channel_w*self.channel_h*self.channel_num     # mm2
        # D_h = 2*self.channel_w*self.channel_h/(self.channel_w+self.channel_h) # mm2
        # u_f = (F_f_tot*Rgas*self._T_in)/(Pf* 1e5 * A_chan)                #mm/s
        # Re_f = rho_f*u_f*D_h/mu_f
        # fric_f = 64/Re_f
        # dPfdz = -fric_f*(rho_f*u_f**2)/(2*D_h) * 1e-5 # bar/mm
        
        # u_p = (F_p_tot*Rgas*self._T_in)/(Pp*1e5 * A_chan)
        # Re_p = rho_p*u_p*D_h/mu_p
        # fric_p = 64/Re_p
        # dPpdz = fric_p*(rho_p*u_p**2)/(2*D_h) * 1e-5 # bar/mm
        
        dPfdz = -12*mu_f/(self.channel_h**3 * self.channel_w)*(F_f_tot/self.channel_num* Rgas * self._T_in)/(Pf*1e5)*1e-5
        # 모든 채널로 동일한 유량 흐름

        dPpdz = 12*mu_p/(self.channel_h**3 * self.channel_w)*(F_p_tot/self.channel_num* Rgas * self._T_in)/(Pp*1e5)*1e-5
        # 맨위, 맨 아래는 유량 다른데..
        
           
        ## 
        J = self._a_perm*(Pf_i - Pp_i) # mol/(mm2 bar s) * bar = mol/(mm2 s)
        arg_neg_J = J < 0
        J[arg_neg_J] = 0
        
        dF_f = -2*self.channel_num*self.channel_w*J     # mm* mol/(mm2 s) = mol/(mm s)
        dF_p = - 2*self.channel_num*self.channel_w*J

        dF_f = dF_f.tolist()
        dF_p = dF_p.tolist()

        dydz = dF_f+ dF_p+ [dPfdz]+[dPpdz]

        return dydz
   
    def initialC_info(self, on=True):
        """Derive (for co-current) or set (for counter-current) initial condition
        """
        
        if self._config[:2] == 'co':
            self._Pp_in = 1.01
            
        elif self._config[:2] == 'ct':
            # Fp_init = np.array([self._Ff_in[ii]*0.5 for ii in range(self._n_comp)])
            self._Pp_in = 1
        
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
        
        if self._config == 'co':
            model = self._channel_co
        elif self._config == 'ct':
            model =  self._channel_ct


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
            _factor = 2 * self.channel_w * self.channel_num

            if self._config[:2] == 'ct':
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
        if self._config[:2] in ['CO', 'co']:
            inpt = sum(y[0,:self._n_comp*2])
            outp = sum(y[-1,:self._n_comp*2])
        elif self._config[:2] in ['CT', 'ct']:
            inpt = sum(y[0,:self._n_comp])+sum(y[-1,self._n_comp:self._n_comp*2])
            outp = sum(y[-1,:self._n_comp])+sum(y[0,self._n_comp:self._n_comp*2])
        
        err = abs(inpt-outp)/inpt*100
        print('Mass balance (error %): ', err)
        return err
    
    
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
        
        if self._config[:2] == 'co':
            dPp = (Pp[0]-Pp)*1e5
        elif self._config[:2] == 'ct':
            dPp = (Pp[-1]-Pp)*1e5

        ########### flux  ##########
        fig = plt.figure(figsize=(10,7),dpi=200)
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
            
        

if __name__ == '__main__':
    F_feed = 10
    A_tot_m2 = 94.24           # m2
    h_channel_um = 250      # um
    a_perm_mol_m_pa = np.array([0.1, 10])*3.35e-10     #Permeance(mol/(m2 Pa s)) --> 오류? (mol/(m2 bar s))
    W_width_m = 0.9424          # m
    L_channel_m = 1             # Channel length (m)
    N_channel = 50
    d_mem_um = 200              # membrane thickness (um)

    y_feed = np.array([0.79, 0.21])     # mole fraction (N2, O2)
    P_feed_pa = 1034000
    P_perm_pa = 103400
    T_feed_K = 298.15
    
    N_node = 100
    # Operating conditions
    P_feed = P_feed_pa*1e-5                # pressure of feed side (bar)

    y_feed = y_feed                        # mole fraction (N2, O2)

    L_channel = L_channel_m*1e3            # Channel length (mm)
    w_channel = W_width_m*1e3              # Channel width (mm)
    h_channel = h_channel_um*1e-3          # Channel height (mm)
    N_channel = N_channel                  # number of channels (-)

    # Gas properties
    Mw_i = np.array([28e-3, 32e-3])     # Molar weight (kg/mol)

    rho_i = np.array([1.17, 1.291])*1e-9     # Density (kg/mm3)
    mu_N2 = 17.82e-6
    mu_O2 = 20.55e-6           # O2 viscosity (Pa s)
    # viscosity values from https://www.engineeringtoolbox.com/gases-absolute-dynamic-viscosity-d_1888.html
    mu_i = np.array([mu_N2, mu_O2])   # (Pa s)
    k_mass =1e-1
    # Constants
    Rgas = 8.314*1e9                # mm3 Pa/K mol

    N = N_node
    a_perm = a_perm_mol_m_pa*1e5*1e-6     #Permeance(mol/(mm2 bar s))

    mem_model = PFM(config = 'co',
                    channel_num=N_channel,
                    channel_size=[L_channel, w_channel, h_channel],       #mm
                    n_component=2,
                    n_node=N)

    mem_model.membrane_info(thickness=d_mem_um*1e-3,              # mm
                            a_perm=a_perm)
    mem_model.gas_prop_info(molar_mass = Mw_i,
                            mu_viscosity = mu_i, 
                            rho_density = rho_i)
    mem_model.mass_trans_info(k_mass_transfer = k_mass)
    mem_model.boundaryC_info(y_inlet = y_feed,
                                p_f_inlet = P_feed,
                                f_f_inlet = F_feed,
                                T_inlet = T_feed_K,
                                f_sweep = False)
    mem_model.initialC_info()
    y_res = mem_model.run_mem()
    mem_model.PlotResults()

    # %%
    mem_model.PlotResults()
# %%
