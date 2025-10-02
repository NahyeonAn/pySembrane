#%%
import numpy as np
from scipy.optimize import minimize

class FlatSheetMembraneModel:
    def __init__(self,
                 dp_pa_init=900,
                 L_m=1,
                 tol=1e-2,
                 max_iter=50,
                 chan_num=100,
                 N_node=100,
                 h_um=250e-6,
                 w_m=0.9424,
                 rho_gas=40.9,
                 Pf=1034000,
                 Pp=103400):
        
        self.dp_pa_init = dp_pa_init
        self.L_m = L_m
        self.tol = tol
        self.max_iter = max_iter
        self.chan_num = chan_num
        self.N_node = N_node
        self.h_um = h_um
        self.w_m = w_m
        self.rho_gas = rho_gas
        self.Pf = Pf
        self.Pp = Pp
        
        self.mu_i = np.array([17.82e-6, 20.55e-6])  # Pa s
        self.rho_kg_m3 = np.array([1.17, 1.291])
        self.a_perm = np.array([0.1, 10])* 3.35e-10
        self.end_cond = np.array([[0., 0.]])
        self.y_feed = np.array([0.79, 0.21])

    def calculate_mu_f(self, y):
        return np.sum(self.mu_i * y)

    def optimize_permeate_composition(self, xf_k):
        def obj_y_perm(x):
            yp_k = np.tile(np.array([x[0], x[1]]), (self.chan_num//2 + 1, 1))
            
            N_r = xf_k.shape[0]  # 50
            N_p = N_r + 1        # 51

            Qp_k = np.zeros((N_p, 2), dtype=np.float64)

            # 위쪽 retentate에서의 permeation (P1 ~ P50)
            Qp_k[1:] += self.a_perm * (self.Pf * xf_k - self.Pp * yp_k[1:])

            # 아래쪽 retentate에서의 permeation (P0 ~ P49)
            Qp_k[:-1] += self.a_perm * (self.Pf * xf_k - self.Pp * yp_k[:-1])

            # Qp_k = self.a_perm * (self.Pf * np.concatenate([self.end_cond, xf_k]) - self.Pp * yp_k) + \
            #        self.a_perm * (self.Pf * np.concatenate([xf_k, self.end_cond]) - self.Pp * yp_k)
            yp_k_pred = Qp_k / np.sum(Qp_k, axis=1, keepdims=True)
            err = np.sum((yp_k_pred - yp_k)**2)
            print(f"Optimized permeate composition, error: {err}")
            return err
        
        result = minimize(obj_y_perm, [self.y_feed[0], self.y_feed[0]], bounds=[(0, 1), (0, 1)], tol=1e-6)
        
        return result.x

    def calculate_Qrk_i(self, Qr_i_init, yp_k_pred):
        Qrk_i = []
        _Qr_i = Qr_i_init.copy()
        for _ in range(self.N_node):
            xr_i = _Qr_i / np.sum(_Qr_i, axis=1, keepdims=True)
            _Qr_i = _Qr_i - (self.w_m * self.L_m/self.N_node) * self.a_perm * (self.Pf * xr_i - self.Pp * yp_k_pred[:1]) \
                              - (self.L_m/self.N_node) * self.a_perm * (self.Pf * xr_i - self.Pp * yp_k_pred[1:])
            Qrk_i.append(_Qr_i)
        return Qrk_i

    def calculate_Pkr_i(self, Qrk_i):
        Pkr_i = []
        P_next = self.Pf
        for Q_stage in Qrk_i:
            x_stage = Q_stage / np.sum(Q_stage, axis=1, keepdims=True)
            mu_f_stage = np.sum(self.mu_i * x_stage, axis=1)
            lambda_stage = 12 * mu_f_stage * (self.L_m / self.N_node) / (self.h_um**3 * self.w_m * self.rho_gas)
            Q_tot_per_channel = np.sum(Q_stage, axis=1)
            dP_stage = lambda_stage * Q_tot_per_channel
            P_next = P_next - dP_stage
            Pkr_i.append(P_next)
        return Pkr_i

    def run(self):
        Qtot_init = self.dp_pa_init / (12 * self.calculate_mu_f(self.y_feed) * self.L_m / 
                                       (self.h_um**3 * self.w_m * self.rho_gas))

        for loop in range(self.max_iter):
            Qr_i_init = np.tile(Qtot_init * self.y_feed, (self.chan_num//2, 1))
            xf_k = np.tile(self.y_feed, (self.chan_num//2, 1))

            yp_k_pred = self.optimize_permeate_composition(xf_k)
            Qrk_i = self.calculate_Qrk_i(Qr_i_init, yp_k_pred)
            Pkr_i = self.calculate_Pkr_i(Qrk_i)

            dP_pred = np.mean(Pkr_i[0]) - np.mean(Pkr_i[-1])
            error = abs(dP_pred - self.dp_pa_init)

            print(f"[{loop+1:2d}] dP_pred = {dP_pred:.2f} Pa, error = {error:.4f}")
            if error < self.tol:
                print("수렴 완료.")
                break

            Qtot_init *= self.dp_pa_init / dP_pred
        else:
            print("최대 반복 횟수 도달. 수렴하지 않음.")

        Qp_k_all = []
        for j in range(self.N_node - 1):
            delta_Q = Qrk_i[j] - Qrk_i[j + 1]  # stage j → j+1 사이의 유량 감소 = permeated
            Qp_k_all.append(delta_Q)
        Qp_total_comp = np.sum(Qp_k_all, axis=(0, 1))  # shape: (2,)
        
        return {
            "Q_tot": Qtot_init*(self.chan_num//2),
            "Qrk_i": Qrk_i,              # retentate flow (chan_num/2, N_node, 2)
            "Pkr_i": Pkr_i,              # retentate pressure
            "Qp_total_comp": Qp_total_comp,  # total permeate composition
            "Y_perm": yp_k_pred,         # permeate mole fraction
            "dP_final": dP_pred
        }
def compute_mass_balance_error(results):
    Q_feed = results['Q_tot']  # 초기 feed 유량 총합 (mol/s)

    # 마지막 stage의 retentate 유량 (list의 마지막 항목을 합산)
    Q_retentate = results['Qrk_i'][-1].sum()

    # permeate 유량 = 각 성분별 유량 총합의 합
    Q_permeate = np.sum(results['Qp_total_comp'])

    # mass balance error
    mass_balance_error = abs(Q_feed - (Q_retentate + Q_permeate)) / Q_feed
    return mass_balance_error

model = FlatSheetMembraneModel(dp_pa_init=900)
results = model.run()

dP_res =[]
for dP in np.logspace(np.log10(900), np.log10(39000), 20):
    model = FlatSheetMembraneModel(dp_pa_init=dP)
    results = model.run()
    
    ret_out = results['Qrk_i'][-1].sum(axis=0)
    xf_out = ret_out / np.sum(ret_out)
    
    stg_cut = results['Qp_total_comp'].sum()/results['Q_tot']
    recovery = ret_out.sum()/ results['Q_tot']
    
    err = compute_mass_balance_error(results)
    dP_res.append([xf_out[1], stg_cut, recovery, err, results['Q_tot']])

# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

COLOR_CODE = ["#2364aa", "#3da5d9", "#73bfb8", "#fec601", "#ea7317"]
import matplotlib.cm as cm
def set_common_style():
        
    mpl.rcParams.update({
        # [폰트 관련]
        'font.family': 'Arial',             # 글꼴 종류
        'font.size': 17,                    # 기본 폰트 크기

        # [축 관련]
        'axes.grid': True,                  # 기본 그리드 표시 여부
        'grid.linestyle': '--',             # 그리드 스타일
        'grid.alpha': 0.5,                  # 그리드 투명도

        # [눈금 설정]
        'xtick.direction': 'in',            # 눈금 안쪽으로
        'ytick.direction': 'in',

        # [범례]
        'legend.frameon': False,            # 범례 박스 제거
        'legend.loc': 'best',               # 범례 위치 자동
        'legend.edgecolor': 'none',         # 범례 테두리 색

        # [Figure 설정]
        'figure.figsize': (6, 4),           # 기본 figure 크기 (inches)
        'figure.dpi': 300,                  # 디스플레이 해상도
        'savefig.dpi': 300,                 # 저장 해상도
        'savefig.transparent': True,        # 저장 시 투명 배경
        'savefig.bbox': 'tight',            # 저장 시 여백 최소화

        # [PDF/PS 설정]
        'pdf.fonttype': 42,                 # 벡터화 폰트 유지
        'ps.fonttype': 42,
        
        "axes.prop_cycle": cycler('color', COLOR_CODE)
    })
set_common_style()

val_arr = np.array(dP_res)
plt.figure(figsize=(8,6), dpi=100)
plt.plot(val_arr[:,0], val_arr[:,1], 'o', label = 'PFM prediction', markersize=8, alpha=0.8)
plt.xlabel('O2 mole fraction in retentate')
plt.ylabel('Stage cut')
# plt.xlim([0,0.2])
plt.show()

plt.figure(figsize=(8,6), dpi=100)
plt.plot(val_arr[:,0], val_arr[:,2], 'o',  label = 'PFM prediction', markersize=8, alpha=0.8)
plt.xlabel('O2 mole fraction in retentate')
plt.ylabel('Recovery')
plt.show()
# %%
import numpy as np

class FlatSheetMembraneModel:
    def __init__(self,
                 dp_pa_init=900,
                 L_m=1,
                 tol=1e-2,
                 max_iter=50,
                 chan_num=100,
                 N_node=100,
                 h_um=250e-6,
                 w_m=0.9424,
                 rho_gas=40.9,
                 Pf=10e5,
                 Pp=1e5):
        
        self.dp_pa_init = dp_pa_init
        self.L_m = L_m
        self.tol = tol
        self.max_iter = max_iter
        self.chan_num = chan_num
        self.N_node = N_node
        self.h_um = h_um
        self.w_m = w_m
        self.rho_gas = rho_gas
        self.Pf = Pf
        self.Pp = Pp
        
        self.mu_i = np.array([17.82e-6, 20.55e-6])  # Pa s
        self.rho_kg_m3 = np.array([1.17, 1.291])
        self.a_perm = np.array([0.1, 10]) * 3.35e-10  # mol/(m2·Pa·s)
        self.y_feed = np.array([0.79, 0.21])

    def calculate_mu_f(self, y):
        return np.sum(self.mu_i * y)

    def calculate_Qrk_i_and_ypk(self, Qr_i_init):
        Qrk_i = []
        yp_k_list = []
        _Qr_i = Qr_i_init.copy()
        dx = self.L_m / self.N_node

        for _ in range(self.N_node):
            xr_k = _Qr_i / _Qr_i.sum(axis=1, keepdims=True)  # (chan_num//2, 2)
            yp_k = np.zeros((self.chan_num//2 + 1, 2))

            for k in range(self.chan_num//2):
                flux_top = self.a_perm * (self.Pf * xr_k[k])
                flux_bottom = self.a_perm * (self.Pf * xr_k[k])
                yp_k[k] += flux_top
                yp_k[k+1] += flux_bottom

            yp_k = yp_k / yp_k.sum(axis=1, keepdims=True)
            yp_k = np.nan_to_num(yp_k)

            Qp_k = np.zeros_like(yp_k)

            Qp_k[1:] += self.a_perm * (self.Pf * xr_k - self.Pp * yp_k[1:])
            Qp_k[:-1] += self.a_perm * (self.Pf * xr_k - self.Pp * yp_k[:-1])

            _Qr_i -= dx * self.w_m * (Qp_k[:-1] + Qp_k[1:])
            Qrk_i.append(_Qr_i.copy())
            yp_k_list.append(yp_k.copy())

        return Qrk_i, yp_k_list[-1]

    def calculate_Pkr_i(self, Qrk_i):
        Pkr_i = []
        P_next = self.Pf
        for Q_stage in Qrk_i:
            x_stage = Q_stage / np.sum(Q_stage, axis=1, keepdims=True)
            mu_f_stage = np.sum(self.mu_i * x_stage, axis=1)
            lambda_stage = 12 * mu_f_stage * (self.L_m / self.N_node) / (self.h_um**3 * self.w_m * self.rho_gas)
            Q_tot_per_channel = np.sum(Q_stage, axis=1)
            dP_stage = lambda_stage * Q_tot_per_channel
            P_next = P_next - dP_stage
            Pkr_i.append(P_next)
        return Pkr_i

    def run(self):
        Qtot_init = self.dp_pa_init / (12 * self.calculate_mu_f(self.y_feed) * self.L_m / 
                                       (self.h_um**3 * self.w_m * self.rho_gas))

        for loop in range(self.max_iter):
            Qr_i_init = np.tile(Qtot_init * self.y_feed, (self.chan_num//2, 1))
            Qrk_i, yp_k_pred = self.calculate_Qrk_i_and_ypk(Qr_i_init)
            Pkr_i = self.calculate_Pkr_i(Qrk_i)

            dP_pred = np.mean(Pkr_i[0]) - np.mean(Pkr_i[-1])
            error = abs(dP_pred - self.dp_pa_init)

            print(f"[{loop+1:2d}] dP_pred = {dP_pred:.2f} Pa, error = {error:.4f}")
            if error < self.tol:
                print("수렴 완료.")
                break

            Qtot_init *= self.dp_pa_init / dP_pred
        else:
            print("최대 반복 횟수 도달. 수렴하지 않음.")

        Qp_k_all = []
        for j in range(self.N_node - 1):
            delta_Q = Qrk_i[j] - Qrk_i[j + 1]
            Qp_k_all.append(delta_Q)

        Qp_total_comp = np.sum(Qp_k_all, axis=(0, 1))
        
        return {
            "Q_tot": Qtot_init * (self.chan_num//2),
            "Qrk_i": Qrk_i,
            "Pkr_i": Pkr_i,
            "Qp_total_comp": Qp_total_comp,
            "Y_perm": yp_k_pred,
            "dP_final": dP_pred
        }



model = FlatSheetMembraneModel(dp_pa_init=900)
results = model.run()

dP_res =[]
for dP in np.logspace(np.log10(900), np.log10(390000), 20):
    model = FlatSheetMembraneModel(dp_pa_init=dP)
    results = model.run()
    
    ret_out = results['Qrk_i'][-1].sum(axis=0)
    xf_out = ret_out / np.sum(ret_out)
    
    stg_cut = results['Qp_total_comp'].sum()/results['Q_tot']
    recovery = ret_out.sum()/ results['Q_tot']
    
    err = compute_mass_balance_error(results)
    dP_res.append([xf_out[1], stg_cut, recovery, err, results['Q_tot']])


# %%
import numpy as np
from scipy.optimize import minimize

class FlatSheetMembraneModel:
    def __init__(self,
                 dp_pa_init=900,
                 L_m=1,
                 tol=1e-2,
                 max_iter=50,
                 chan_num=100,
                 N_node=100,
                 h_um=250e-6,
                 w_m=0.9424,
                 rho_gas=40.9,
                 Pf=10e5,
                 Pp=1e5):

        self.dp_pa_init = dp_pa_init
        self.L_m = L_m
        self.tol = tol
        self.max_iter = max_iter
        self.chan_num = chan_num
        self.N_node = N_node
        self.h_um = h_um
        self.w_m = w_m
        self.rho_gas = rho_gas
        self.Pf = Pf
        self.Pp = Pp

        self.mu_i = np.array([17.82e-6, 20.55e-6])  # Pa s
        self.rho_kg_m3 = np.array([1.17, 1.291])
        self.a_perm = np.array([0.1, 10]) * 3.35e-10  # GPU to mol/m2/s/Pa
        self.y_feed = np.array([0.79, 0.21])

    def calculate_mu_f(self, y):
        return np.sum(self.mu_i * y)

    def calculate_pressure_profile(self, Qrk_i):
        Pkr_i = []
        P_next = self.Pf
        for Q_stage in Qrk_i:
            x_stage = Q_stage / np.sum(Q_stage, axis=1, keepdims=True)
            mu_f_stage = np.sum(self.mu_i * x_stage, axis=1)
            lambda_stage = 12 * mu_f_stage * (self.L_m / self.N_node) / (
                self.h_um**3 * self.w_m * self.rho_gas)
            Q_tot_per_channel = np.sum(Q_stage, axis=1)
            dP_stage = lambda_stage * Q_tot_per_channel
            P_next = P_next - dP_stage
            Pkr_i.append(P_next)
        return np.array(Pkr_i)

    def run(self):
        Qtot_init = self.dp_pa_init / (12 * self.calculate_mu_f(self.y_feed) * self.L_m /
                                       (self.h_um**3 * self.w_m * self.rho_gas))

        for loop in range(self.max_iter):
            Qr_i = np.tile(Qtot_init * self.y_feed, (self.chan_num // 2, 1))
            Qrk_i = [Qr_i.copy()]

            for _ in range(self.N_node - 1):
                xr_i = Qr_i / np.sum(Qr_i, axis=1, keepdims=True)
                Pkr_i = self.calculate_pressure_profile(Qrk_i)
                P_stage = Pkr_i[-1]  # (chan_num/2,)
                # permeation flux: J = a_perm * (P_r * x - P_p * y_perm)
                # permeate 조성은 feed와 같다고 가정
                y_perm = xr_i.copy()
                J_i = self.a_perm * (P_stage[:, None] * xr_i - self.Pp * y_perm)

                dQ = self.w_m * self.L_m / self.N_node * J_i
                Qr_i = Qr_i - dQ
                Qrk_i.append(Qr_i.copy())

            Pkr_i = self.calculate_pressure_profile(Qrk_i)
            dP_pred = np.mean(Pkr_i[0]) - np.mean(Pkr_i[-1])
            error = abs(dP_pred - self.dp_pa_init)

            print(f"[{loop+1:2d}] dP_pred = {dP_pred:.2f}, error = {error:.5f}")
            if error < self.tol:
                print("수렴 완료.")
                break

            Qtot_init *= self.dp_pa_init / dP_pred

        Qp_k_all = []
        for j in range(self.N_node - 1):
            delta_Q = Qrk_i[j] - Qrk_i[j + 1]
            Qp_k_all.append(delta_Q)
        Qp_total_comp = np.sum(Qp_k_all, axis=(0, 1))

        return {
            "Q_tot": Qtot_init * (self.chan_num // 2),
            "Qrk_i": Qrk_i,
            "Pkr_i": Pkr_i,
            "Qp_total_comp": Qp_total_comp,
            "dP_final": dP_pred
        }


def compute_mass_balance_error(results):
    Q_feed = results['Q_tot']
    Q_retentate = results['Qrk_i'][-1].sum()
    Q_permeate = np.sum(results['Qp_total_comp'])
    mass_balance_error = abs(Q_feed - (Q_retentate + Q_permeate)) / Q_feed
    return mass_balance_error

model = FlatSheetMembraneModel(dp_pa_init=900)
results = model.run()

dP_res =[]
for dP in np.logspace(np.log10(900), np.log10(390000), 20):
    model = FlatSheetMembraneModel(dp_pa_init=dP)
    results = model.run()
    
    ret_out = results['Qrk_i'][-1].sum(axis=0)
    xf_out = ret_out / np.sum(ret_out)
    
    stg_cut = results['Qp_total_comp'].sum()/results['Q_tot']
    recovery = ret_out.sum()/ results['Q_tot']
    
    err = compute_mass_balance_error(results)
    dP_res.append([xf_out[1], stg_cut, recovery, err, results['Q_tot']])

val_arr = np.array(dP_res)
plt.figure(figsize=(8,6), dpi=100)
plt.plot(val_arr[:,0], val_arr[:,1], 'o', label = 'PFM prediction', markersize=8, alpha=0.8)
plt.xlabel('O2 mole fraction in retentate')
plt.ylabel('Stage cut')
# plt.xlim([0,0.2])
plt.show()

plt.figure(figsize=(8,6), dpi=100)
plt.plot(val_arr[:,0], val_arr[:,2], 'o',  label = 'PFM prediction', markersize=8, alpha=0.8)
plt.xlabel('O2 mole fraction in retentate')
plt.ylabel('Recovery')
plt.show()