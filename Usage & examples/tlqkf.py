#%%
import numpy as np
dp_pa_init = 900
L_m = 1

mu_N2 = 17.82e-6
mu_O2 = 20.55e-6           # O2 viscosity (Pa s)
# viscosity values from https://www.engineeringtoolbox.com/gases-absolute-dynamic-viscosity-d_1888.html
mu_i = np.array([mu_N2, mu_O2])   # (Pa s)
y_feed = np.array([0.79, 0.21])     # mole fraction (N2, O2)
mu_f = np.sum(mu_i*y_feed)

h_um = 250e-6  # channel height (m)
w_m = 0.9424  # channel width (m)
rho_kg_m3 = np.array([1.17, 1.291])
rho_f = np.sum(rho_kg_m3 * y_feed)  # kg/m3
rho_gas = 40.9  # mol/m3   --> Q 가 mol/s 이므로, rho_gas 는 mol density

Qtot_init = dp_pa_init/(12*mu_f*L_m/(h_um**3*w_m*rho_gas))    
# print(f"Qtot_init: {Qtot_init} mol/s")

# Mw_i = np.array([28e-3, 32e-3])     # Molar weight (kg/mol)
# Mw_f = np.sum(Mw_i * y_feed)  # kg/mol
# Qtot_init = dp_pa_init/(12*mu_f*L_m/(h_um**3*w_m*rho_f))* Mw_f
# print(f"Qtot_init: {Qtot_init} mol/s")
#%%

from scipy.optimize import minimize

chan_num = 100
Qr_i_init = np.tile(Qtot_init * y_feed, (int(chan_num/2), 1))
xf_k = np.tile(y_feed, (int(chan_num/2), 1))

a_perm = np.array([0.1, 10])*3.35e-10  # Permeance (mol/(m2 Pa s)) --> 오류? (mol/(m2 bar s))
Pf = 10e5  # Pa
Pp = 1e5  # Pa
end_cond = np.array([[0., 0.]])
# solve ode
def obj_y_perm(x):
    x_N2, x_O2 = x
    yp_k = np.tile(np.array([x_N2, x_O2]), (int(chan_num/2)+1, 1))

    
    Qp_k =a_perm*(Pf*np.concatenate([end_cond, xf_k]) - Pp*yp_k) + a_perm*(Pf*np.concatenate([xf_k, end_cond]) - Pp*yp_k)
            # upper                                 # lower
            
    yp_k_pred = Qp_k / np.sum(Qp_k, axis=1, keepdims=True)
    return np.sum(np.abs(yp_k_pred - yp_k)**2)

initial_guess = [y_feed[0], y_feed[0]]  
bounds = [(0, 1), (0, 1)] 
result = minimize(obj_y_perm, initial_guess, bounds=bounds)
yp_k_pred = result.x

Qp_k =a_perm*(Pf*np.concatenate([end_cond, xf_k]) - Pp*yp_k_pred) + a_perm*(Pf*np.concatenate([xf_k, end_cond]) - Pp*yp_k_pred)
#%%
Qrk_i = []
N_node = 100
_Qr_i =  Qr_i_init.copy()
for j in range(N_node):
    xr_i = _Qr_i / np.sum(_Qr_i, axis=1, keepdims=True)
    _Qr_i = _Qr_i - L_m/N_node*a_perm*(Pf*xr_i-Pp*yp_k_pred[:1])- L_m/N_node*a_perm*(Pf*xr_i-Pp*yp_k_pred[1:])
    Qrk_i.append(_Qr_i)

lambda_ = 12 * mu_f * L_m / (h_um**3 * w_m * rho_gas)
Pkr_i = []
P_next = Pf
for j in range(N_node):
    Q_tot_stage = np.sum(Qrk_i[j], axis=1)  # shape: (chan_num/2, )
    dP_stage = lambda_ * Q_tot_stage * (L_m / N_node)  # ΔP per channel at stage j
    P_next = P_next - dP_stage  # 평균 ΔP로 압력 업데이트
    Pkr_i.append(P_next)
dP_pred = Pkr_i[0][0] - Pkr_i[-1][0]  # 마지막 단계와 첫 번째 단계의 압력 차이
Qr_i_new = Qr_i_init*(dp_pa_init/dP_pred)

# x_p_init = y_feed
# yp_k = np.tile(x_p_init, (int(chan_num/2)+1, 1))
# Qp_init = Qr_i_init * x_p_init


# Pf = 10e5  # Pa
# Pp = 1e5  # Pa
# a_perm = np.array([0.1, 10])

# end_cond = np.array([[0., 0.]])
# Qp_k =a_perm*(Pf*np.concatenate([end_cond, xf_k]) - Pp*yp_k) + a_perm*(Pf*np.concatenate([xf_k, end_cond]) - Pp*yp_k)
#         # upper                                 # lower
        
# yp_k_pred = Qp_k / np.sum(Qp_k, axis=1, keepdims=True)

# %%
import numpy as np
from scipy.optimize import minimize

# 초기 설정
dp_pa_init = 900
L_m = 1
tol = 1e-2  # 수렴 기준
max_iter = 50

mu_N2 = 17.82e-6        # Pa s
mu_O2 = 20.55e-6
mu_i = np.array([mu_N2, mu_O2])
y_feed = np.array([0.79, 0.21])
mu_f = np.sum(mu_i * y_feed)

h_um = 250e-6
w_m = 0.9424
rho_kg_m3 = np.array([1.17, 1.291])
rho_f = np.sum(rho_kg_m3 * y_feed)
rho_gas = 40.9  # mol/m3

# 초기 유량 계산
Qtot_init = dp_pa_init / (12 * mu_f * L_m / (h_um**3 * w_m * rho_gas))

chan_num = 100
N_node = 100
end_cond = np.array([[0., 0.]])
a_perm = np.array([0.1, 10]) * 3.35e-10
Pf = 10e5
Pp = 1e5

lambda_ = 12 * mu_f * L_m / (h_um**3 * w_m * rho_gas)

# 압력 강하 수렴 루프
for loop in range(max_iter):
    # 1. 초기 유량 및 조성
    Qr_i_init = np.tile(Qtot_init * y_feed, (chan_num//2, 1))
    xf_k = np.tile(y_feed, (chan_num//2, 1))

    # 2. permeate 조성 최적화
    def obj_y_perm(x):
        x_N2, x_O2 = x
        yp_k = np.tile(np.array([x_N2, x_O2]), (chan_num//2 + 1, 1))
        Qp_k = a_perm * (Pf * np.concatenate([end_cond, xf_k]) - Pp * yp_k) + \
               a_perm * (Pf * np.concatenate([xf_k, end_cond]) - Pp * yp_k)
        yp_k_pred = Qp_k / np.sum(Qp_k, axis=1, keepdims=True)
        return np.sum((yp_k_pred - yp_k)**2)

    result = minimize(obj_y_perm, [y_feed[0], y_feed[0]], bounds=[(0, 1), (0, 1)])
    yp_k_pred = result.x

    # 3. Qr 계산 (stage loop)
    Qrk_i = []
    _Qr_i = Qr_i_init.copy()
    for j in range(N_node):
        xr_i = _Qr_i / np.sum(_Qr_i, axis=1, keepdims=True)
        _Qr_i = _Qr_i - (L_m/N_node) * a_perm * (Pf * xr_i - Pp * yp_k_pred[:1]) \
                          - (L_m/N_node) * a_perm * (Pf * xr_i - Pp * yp_k_pred[1:])
        Qrk_i.append(_Qr_i)

    # 4. 압력 계산
    Pkr_i = []
    P_next = Pf
    for j in range(N_node):
        Q_stage = Qrk_i[j]  # shape: (chan_num/2, 2)
        x_stage = Q_stage / np.sum(Q_stage, axis=1, keepdims=True)  # (chan_num/2, 2)
        mu_f_stage = np.sum(mu_i * x_stage, axis=1)  # 조성 기반 평균 점도 (chan_num/2,)
        lambda_stage = 12 * mu_f_stage * (L_m / N_node) / (h_um**3 * w_m * rho_gas)
        
        Q_tot_stage = np.sum(Q_stage, axis=1)  # 각 채널별 유량 (chan_num/2,)
        dP_stage = lambda_stage * Q_tot_stage  # ΔP per channel at stage j
        P_next = P_next - dP_stage  # 압력 업데이트
        Pkr_i.append(P_next)

    # 5. 압력 강하 계산 및 수렴 체크
    dP_pred = np.mean(Pkr_i[0]) - np.mean(Pkr_i[-1])
    error = abs(dP_pred - dp_pa_init)/dp_pa_init*100

    print(f"[{loop+1:2d}] dP_pred = {dP_pred:.2f} Pa, error = {error:.4f}")

    if error < tol:
        print("수렴 완료.")
        break

    # 6. 유량 업데이트
    Qtot_init *= dp_pa_init / dP_pred

else:
    print("최대 반복 횟수 도달. 수렴하지 않음.")
