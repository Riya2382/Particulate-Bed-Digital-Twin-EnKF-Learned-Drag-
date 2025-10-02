
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def drag_model(ug, eps_s, C0=2.0):
    return C0  # baseline constant drag (misspecified on purpose)

def step_model(state, ug, dt, C0=2.0, dCd=0.0):
    eps_s, H = state
    Cd = drag_model(ug, eps_s, C0) + dCd
    deps = dt*( -0.6*(eps_s-0.5) + 0.2*(ug-0.8) - 0.1*Cd*(eps_s-0.45) )
    dH  = dt*(  0.3*(ug-0.8) - 0.2*(eps_s-0.5) )
    eps_s = np.clip(eps_s + deps, 0.3, 0.65)
    H = np.clip(H + dH, 0.7, 1.3)
    return np.array([eps_s, H])

def enkf_filter(ug, y_eps, y_H, dt, N=40, C0=2.0, R=1e-4, Q=1e-5):
    T = len(ug)
    ens = np.tile(np.array([0.5,1.0]), (N,1)) + 0.01*np.random.randn(N,2)
    xs = []
    for t in range(T):
        # forecast
        for i in range(N):
            ens[i] = step_model(ens[i], ug[t], dt, C0=C0, dCd=0.0) + np.random.randn(2)*np.sqrt(Q)
        x_pred = ens.mean(0)
        P = np.cov(ens.T) + 1e-6*np.eye(2)

        # observe
        Hmat = np.eye(2)  # we observe both
        y = np.array([y_eps[t], y_H[t]])
        Rmat = R*np.eye(2)
        K = P @ Hmat.T @ np.linalg.inv(Hmat @ P @ Hmat.T + Rmat)
        # update each ensemble member with perturbed obs
        for i in range(N):
            y_pert = y + np.random.randn(2)*np.sqrt(R)
            ens[i] = ens[i] + K @ (y_pert - Hmat @ ens[i])
        xs.append(ens.mean(0))
    return np.array(xs)

def main():
    data = np.load("outputs/truth.npz")
    ug, y_eps, y_H, dt = data["ug"], data["y_eps"], data["y_H"], float(data["dt"])
    x_est = enkf_filter(ug, y_eps, y_H, dt, N=40)

    out = Path("outputs"); out.mkdir(parents=True, exist_ok=True)
    np.savez(out/"enkf.npz", x_est=x_est)
    # plots
    plt.figure(); plt.plot(data["eps_s"],label="truth"); plt.plot(x_est[:,0],label="enkf"); plt.legend(); plt.title("eps_s"); plt.xlabel("t"); plt.ylabel("eps_s")
    plt.savefig(out/"enkf_eps.png", dpi=150); plt.close()
    plt.figure(); plt.plot(data["H"],label="truth"); plt.plot(x_est[:,1],label="enkf"); plt.legend(); plt.title("H"); plt.xlabel("t"); plt.ylabel("H")
    plt.savefig(out/"enkf_H.png", dpi=150); plt.close()
    print("Saved EnKF results to outputs/enkf.npz")

if __name__=="__main__":
    main()
