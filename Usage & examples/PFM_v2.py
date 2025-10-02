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
class PFM:
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
    
    def _unpack_state(self, y):
        
        N = self.channel_num        # retentate 채널 수
        N_p = N + 1                 # permeate 채널 수
        C = self._n_comp            # 성분 수

        # 상태벡터 y 길이 = N*2C + N + N_p
        F_f = [np.array(y[i*C:(i+1)*C]) for i in range(N)]
        F_p = [np.array(y[N*C + i*C : N*C + (i+1)*C]) for i in range(N_p)]
   
        pf_start = N*C + N_p*C
        pf_end = pf_start + N
        pp_start = pf_end

        Pf = np.array(y[pf_start : pf_end])
        Pp = np.array(y[pp_start : ])

        return F_f, F_p, Pf, Pp

    def _channel_co_test(self, y, z):
        n, nc = self.channel_num, self._n_comp
        T = self._T_in
        h, w = self.channel_h, self.channel_w
        
        # 상태 변수 분해
        Ff_list, Fp_list, Pf, Pp = self._unpack_state(y)
        
        dydz = []
        dFp_list = [np.zeros(nc) for _ in range(n+1)]
        for i in range(n):

            Ff_i = Ff_list[i]
            Pf_i = Pf[i]

            x_i = Ff_i / np.sum(Ff_i)
            Pf_comp = x_i * Pf_i
            # 각각의 양쪽 permeate 조성/압력
            Pp_left = Pp[i]
            Pp_right = Pp[i+1]
            y_left = Fp_list[i]   / np.sum(Fp_list[i])
            y_right = Fp_list[i+1]/ np.sum(Fp_list[i+1])
            Pp_comp_left = y_left * Pp_left

            Pp_comp_right = y_right * Pp_right
            J_left = self._a_perm * (Pf_comp - Pp_comp_left)
            J_right = self._a_perm * (Pf_comp - Pp_comp_right)

            # 음의 flux 제거
            J_left[J_left < 0] = 0
            J_right[J_right < 0] = 0
            
            # Feed 쪽 mol flow 감소
            dFf = -w * J_left  - w * J_right   # membrane 면적 고려
            dydz.extend(dFf.tolist())

            dFp_list[i]   += w * J_left
            dFp_list[i+1] += w * J_right
            
        for dFp in dFp_list:             # 각 permeate 채널의 유량 변화 저장
            dydz.extend(dFp.tolist())
        
        for i in range(n):
            Ff_i = Ff_list[i]
            Pf_i = Pf[i]
            x_i = Ff_i / np.sum(Ff_i)
            mu_f = np.sum(self._mu * x_i)

            dPf = -12 * mu_f / (h**3 * w) * (np.sum(Ff_i) * Rgas * T) / (Pf_i * 1e5) * 1e-5
            dydz.append(dPf)
            
        for i in range(n+1):
            Fp_i = Fp_list[i]
            Pp_i = Pp[i]
            y_i = Fp_i / np.sum(Fp_i)
            mu_p = np.sum(self._mu * y_i)

            dPp = -12 * mu_p / (h**3 * w) * (np.sum(Fp_i) * Rgas * T) / (Pp_i * 1e5) * 1e-5
            dydz.append(dPp)
            
        return dydz

    def _channel_ct_test(self, y, z):
        n, nc = self.channel_num, self._n_comp
        T = self._T_in
        h, w = self.channel_h, self.channel_w
        
        # 상태 변수 분해
        Ff_list, Fp_list, Pf, Pp = self._unpack_state(y)
        
        dydz = []
        dFp_list = [np.zeros(nc) for _ in range(n+1)]
        for i in range(n):

            Ff_i = Ff_list[i]
            Pf_i = Pf[i]

            x_i = Ff_i / np.sum(Ff_i)
            Pf_comp = x_i * Pf_i
            # 각각의 양쪽 permeate 조성/압력
            Pp_left = Pp[i]
            Pp_right = Pp[i+1]
            y_left = Fp_list[i]   / np.sum(Fp_list[i])
            y_right = Fp_list[i+1]/ np.sum(Fp_list[i+1])
            Pp_comp_left = y_left * Pp_left

            Pp_comp_right = y_right * Pp_right
            J_left = self._a_perm * (Pf_comp - Pp_comp_left)
            J_right = self._a_perm * (Pf_comp - Pp_comp_right)

            # 음의 flux 제거
            J_left[J_left < 0] = 0
            J_right[J_right < 0] = 0
            
            # Feed 쪽 mol flow 감소
            dFf = -w * J_left  - w * J_right   # membrane 면적 고려
            dydz.extend(dFf.tolist())

            dFp_list[i]   -= w * J_left
            dFp_list[i+1] -= w * J_right
            
        for dFp in dFp_list:             # 각 permeate 채널의 유량 변화 저장
            dydz.extend(dFp.tolist())
        
        for i in range(n):
            Ff_i = Ff_list[i]
            Pf_i = Pf[i]
            x_i = Ff_i / np.sum(Ff_i)
            mu_f = np.sum(self._mu * x_i)

            dPf = -12 * mu_f / (h**3 * w) * (np.sum(Ff_i) * Rgas * T) / (Pf_i * 1e5) * 1e-5
            dydz.append(dPf)
            
        for i in range(n+1):
            Fp_i = Fp_list[i]
            Pp_i = Pp[i]
            y_i = Fp_i / np.sum(Fp_i)
            mu_p = np.sum(self._mu * y_i)

            dPp = 12 * mu_p / (h**3 * w) * (np.sum(Fp_i) * Rgas * T) / (Pp_i * 1e5) * 1e-5
            dydz.append(dPp)
            
        return dydz
    
    def initialC_info(self, on=True, Pp_in = None):
        """Derive (for co-current) or set (for counter-current) initial condition
        """
        
        if self._config == 'co':
            self._Pp_in = 1.01
            
        elif self._config == 'ct':
            self._Pp_in = 1
        
        if Pp_in:
            self._Pp_in = Pp_in

        n, nc = self.channel_num, self._n_comp

        # 초기 retentate 조성 및 유량
        Ff0 = np.array(self._Ff_in)/n  # shape = (nc,)
        Ff_init = [Ff0.copy() for _ in range(n)]  # 동일한 유량으로 초기화

        # 초기 permeate 유량: 0으로 시작
        Fp_init = [np.array([0.5,0.5])*1e-7 for _ in range(n + 1)]

        # 초기 압력
        Pf_init = [self._Pf_in for _ in range(n)]      # 예: 10 bar
        Pp_init = [self._Pp_in for _ in range(n + 1)]  # 예: 1 bar (vacuum)

        # Flatten into single y vector
        y0 = []
        for Ff in Ff_init:
            y0.extend(Ff.tolist())
        for Fp in Fp_init:
            y0.extend(Fp.tolist())
        y0.extend(Pf_init)
        y0.extend(Pp_init)
        
        self._y0 = np.array(y0)
        self._required['InitialC_info'] = True


    # def _CalculCP(self, Pf, Pp, x_f, y_p):
    #     A_vol = self._a_perm*self._molar_mass/self._rho      # Volumetric permeance
    #     k = self._k_mtc
    #     n_comp = self._n_comp
    #     P_ref, T_ref = self._cp_cond
    #     X = self._Pf_in/P_ref*T_ref/self._T_in
    #     M_mtr = np.zeros((n_comp-1, n_comp-1))
    #     for jj in range(n_comp-1):
    #         for ii in range(n_comp-1):
    #             if ii == jj:
    #                 M_mtr[ii, jj] = A_vol[ii]*Pf+k*X-x_f[ii]*Pf*(A_vol[ii]-A_vol[-1])
    #             else:
    #                 M_mtr[ii, jj] = -x_f[jj]*Pf*(A_vol[ii]-A_vol[-1])
        
    #     M_inv = np.linalg.inv(M_mtr)
    #     sum_y = np.sum([A_vol[ii]*y_p[ii] for ii in range(n_comp-1)])
    #     Y = A_vol[-1]*Pf-A_vol[-1]*Pp*y_p[-1]-Pp*sum_y
    #     b = [A_vol[ii]*Pp*y_p[ii]+k*X*x_f[ii]+x_f[ii]*Y for ii in range(n_comp-1)]

    #     x_n_1 = np.dot(M_inv,b)
    #     x_mem = np.insert(x_n_1, n_comp-1, 1-sum(x_n_1))
    #     return x_mem

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
            model = self._channel_co_test
        elif self._config == 'ct':
            model =  self._channel_ct_test
        
        for ii in range(iteration):
            y_res = SolveFDM(model, self._y0, self._z,)
            
            # data extraction
            n_z = y_res.shape[0]
            C = self._n_comp
            Nf = self.channel_num            # retentate 채널 수
            Np = Nf+1
            
            F_f = y_res[:, :C*Nf]
            F_p = y_res[:, C*Nf:C*(Nf+Np)]
            Pf = y_res[:, C*(Nf+Np):C*(Nf+Np)+Nf] 
            Pp = y_res[:, C*(Nf+Np)+Nf:]
            
            fp_0_i = F_p[0, :]
            
            J_list = [np.zeros([self._n_node+1, C]) for _ in range(Nf+1)]
            
            # Flux 계산
            for i in range(Nf):
                Ff_i = F_f[:,i*C:(i+1)*C]
                Fp_i_left = F_p[:,i*C:(i+1)*C]
                Fp_i_right = F_p[:,(i+1)*C:(i+2)*C]
                
                x_i = Ff_i / np.sum(Ff_i, axis=1, keepdims=True)
                Pf_i = Pf[:,[i]]
                Pf_comp = x_i * Pf_i 
                
                Pp_i_left = Pp[:,[i]]
                Pp_i_right = Pp[:,[i+1]]

                y_left = Fp_i_left / np.sum(Fp_i_left, axis=1, keepdims=True)
                y_right = Fp_i_right / np.sum(Fp_i_right, axis=1, keepdims=True)
                
                Pp_comp_left = y_left * Pp_i_left
                Pp_comp_right = y_right * Pp_i_right
                
                J_left = self._a_perm * (Pf_comp - Pp_comp_left)
                J_right = self._a_perm * (Pf_comp - Pp_comp_right)

                # 음의 flux 제거
                J_left[J_left < 0] = 0
                J_right[J_right < 0] = 0
                
                A_mem = self.channel_L/self._n_node*self.channel_w

                J_list[i]   += A_mem * J_left
                J_list[i+1] += A_mem * J_right

            if self._config == 'co':
                err = [(fp_0_i[ch*C:(ch+1)*C]-J_list[ch][0,:])/fp_0_i[ch*C:(ch+1)*C] for ch in range(Np)]
                err_pp = (1-Pp[-1,:])/Pp[-1,:]
                err.append(err_pp)
            elif self._config == 'ct':
                err = [(fp_0_i[ch*C:(ch+1)*C]-np.sum(J_list[ch][:,:], axis=0))/fp_0_i[ch*C:(ch+1)*C] for ch in range(Np)]
            
            tol = sum([sum(abs(_err)) for _err in err])
            Kg = 0.1
            for ch, _err in enumerate(err):
                if ch < Np:
                    fp_0_i[ch*C:(ch+1)*C] = fp_0_i[ch*C:(ch+1)*C]-Kg*_err*fp_0_i[ch*C:(ch+1)*C]
                else:
                    self._Pp_in = self._Pp_in + Kg*_err*Pp[-1,:]
                    self._y0[C*(Nf+Np)+Nf:] = [_pp for _pp in self._Pp_in]

            self._y0[C*Nf:C*(Nf+Np)] = [_fp for _fp in fp_0_i]

            tolerance = 1e-7
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
        z_dom = (self._z)

        N_f = self.channel_num
        N_p = N_f + 1                     # permeate 채널 수
        C = self._n_comp
        
        _F_f = y_plot[:,:C*N_f]
        _F_p = y_plot[:,C*N_f:C*(N_f+N_p)]
        _Pf = y_plot[:,C*(N_f+N_p):C*(N_f+N_p)+N_f]
        _Pp = y_plot[:,C*(N_f+N_p)+N_f:]
        
        F_f_reshaped = _F_f.reshape(-1, N_f, C)         # shape: (time, N_f, C)
        F_f_by_comp = F_f_reshaped.transpose(2, 1, 0)   # shape: (C, N_f, time)
        F_f = [f_i for f_i in F_f_by_comp]              # list of (N_f, time) arrays

        F_p_reshaped = _F_p.reshape(-1, N_p, C)         # shape: (time, N_f, C)
        F_p_by_comp = F_p_reshaped.transpose(2, 1, 0)   # shape: (C, N_f, time)
        F_p = [p_i for p_i in F_p_by_comp]              # list of (N_f, time) arrays
        
        F_f_total = [F_f[i].sum(axis=0) for i in range(C)]
        F_p_total = [F_p[i].sum(axis=0) for i in range(C)]
        
        J_list = [np.zeros([self._n_node+1, C]) for _ in range(N_f+1)]
        
        # Flux 계산
        for i in range(N_f):
            Ff_i = _F_f[:,i*C:(i+1)*C]
            Fp_i_left = _F_p[:,i*C:(i+1)*C]
            Fp_i_right = _F_p[:,(i+1)*C:(i+2)*C]
            
            x_i = Ff_i / np.sum(Ff_i, axis=1, keepdims=True)
            Pf_i = _Pf[:,[i]]
            Pf_comp = x_i * Pf_i 
            
            Pp_i_left = _Pp[:,[i]]
            Pp_i_right = _Pp[:,[i+1]]

            y_left = Fp_i_left / np.sum(Fp_i_left, axis=1, keepdims=True)
            y_right = Fp_i_right / np.sum(Fp_i_right, axis=1, keepdims=True)
            
            Pp_comp_left = y_left * Pp_i_left
            Pp_comp_right = y_right * Pp_i_right
            
            J_left = self._a_perm * (Pf_comp - Pp_comp_left)
            J_right = self._a_perm * (Pf_comp - Pp_comp_right)

            # 음의 flux 제거
            J_left[J_left < 0] = 0
            J_right[J_right < 0] = 0
            
            J_list[i]   += J_left
            J_list[i+1] += J_right
        
        J_sum = np.sum(np.array(J_list), axis=0)

        ########### flux  ##########
        fig = plt.figure(figsize=(10,7),dpi=200)
        fig.subplots_adjust(hspace=0.5, wspace=0.3)
        ax1 = fig.add_subplot(221)
        for i in range(self._n_comp):
            ax1.plot((self._z*1e-3), (J_sum[:,i]*1e6), linewidth=2,color = c_list[0], 
                        linestyle= line_list[i], label=f'J$_{component[i]}$')
        ax1.set_xlabel('z (m)')
        ax1.set_ylabel('Fluxes [mol/(m$^2$ $\cdot$ s)]')
        ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax1.legend(fontsize=13, loc='best')
        # plt.xlim([0, z_dom[-1]*1e-3])
        if z_ran:
            plt.xlim(z_ran)
        ax1.grid(linestyle='--')
        
        
        ########### Flowrate  ##########
        ax2 = fig.add_subplot(222)
        for i in range(C):
            ax2.plot(z_dom*1e-3, F_f_total[i], linewidth=2,color = c_list[0],
                        linestyle= line_list[i], label=f'Feed$_{component[i]}$')
        ax2.set_xlabel('z (m)')
        ax2.set_ylabel('Feed flowrate (mol/s)')
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax2.grid(linestyle='--')
        ax3 = ax2.twinx()
        for i in range(C):
            ax3.plot(z_dom*1e-3, F_p_total[i], linewidth=2,color = c_list[1],
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

        
        x_i = np.array([F_f_total[i]/np.sum(F_f_total, axis=0) for i in range(C)])
        y_i = np.array([F_p_total[i]/np.sum(F_p_total, axis=0) for i in range(C)])
        
        ########### Mole fraction ##########
        ax4 = fig.add_subplot(223)
        for i in range(C):
            ax4.plot(z_dom*1e-3, x_i[i], linewidth=2,color = c_list[0],
                        linestyle= line_list[i], label=f'Feed$_{component[i]}$')
        ax4.set_xlabel('z (m)')
        plt.ylim([0, 1]) 
        ax4.set_ylabel('Mole fraction (mol/mol)')
        ax4.grid(linestyle='--')
        ax5 = ax4.twinx()
        for i in range(C):
            ax5.plot(z_dom*1e-3, y_i[i], linewidth=2,color = c_list[1],
                        linestyle= line_list[i], label=f'Perm$_{component[i]}$')
        plt.ylim([-0.01, 1.01])    
        if z_ran:
            plt.xlim(z_ran)
        ax4.yaxis.label.set_color(c_list[0])
        ax5.yaxis.label.set_color(c_list[1])
        ax4.tick_params(axis='y', colors=c_list[0])
        ax5.tick_params(axis='y', colors=c_list[1])
        ax5.spines["right"].set_edgecolor(c_list[1])
        ax5.spines["left"].set_edgecolor(c_list[0])
        
        Pf = np.mean(_Pf, axis=1)
        Pp = np.mean(_Pp, axis=1)
        if self._config == 'co':
            dPp = (Pp[0]-Pp)*1e5
            # dPp = (Pp-1)*1e5
        elif self._config == 'ct':
            dPp = (Pp[-1]-Pp)*1e5
            # dPp = (Pp-1)*1e5
        
        ########### Pressure drop ##########
        ax6 = fig.add_subplot(224)
        ax6.plot(z_dom*1e-3, (self._Pf_in-Pf)*1e5, linewidth=2,color = c_list[0],
                    label=f'Feed$_{component[i]}$')
        ax6.set_xlabel('z (m)')
        ax6.set_ylabel('$\Delta$ P$_{f}$ (Pa)')
        # ax6.ticklabel_format(axis='y', style='plain')
        ax6.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax6.grid(linestyle='--')
        ax7= ax6.twinx()
        ax7.plot(z_dom*1e-3, (Pp-1)*1e5, linewidth=2,color = c_list[1],
                        label=f'Perm$_{component[i]}$')
        ax7.set_ylabel('$\Delta$ P$_{p}$ (Pa)')
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
        plt.tight_layout()
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
