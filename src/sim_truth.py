
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def drag_true(ug, eps_s):
    # nonlinear 'true' drag coeff (toy)
    return 2.0 + 1.5*np.tanh(3*(ug-0.8)) + 0.5*np.sin(2*np.pi*eps_s)

def simulate(T=300, dt=0.1, seed=0):
    rng = np.random.RandomState(seed)
    ug = 0.8 + 0.2*np.sin(2*np.pi*np.arange(T)*dt/4.0) + 0.05*rng.randn(T)  # gas superficial velocity
    eps_s = 0.5  # solids holdup (volume fraction)
    H = 1.0      # bed height (normalized)
    eps_s_hist=[]; H_hist=[]
    for t in range(T):
        Cd = drag_true(ug[t], eps_s)
        deps = dt*( -0.6*(eps_s-0.5) + 0.2*(ug[t]-0.8) - 0.1*Cd*(eps_s-0.45) )
        dH  = dt*(  0.3*(ug[t]-0.8) - 0.2*(eps_s-0.5) )
        eps_s = np.clip(eps_s + deps, 0.3, 0.65)
        H = np.clip(H + dH, 0.7, 1.3)
        eps_s_hist.append(eps_s); H_hist.append(H)
    eps_s_hist = np.array(eps_s_hist); H_hist=np.array(H_hist)
    # sensor measurements with noise
    y_eps = eps_s_hist + 0.01*rng.randn(T)
    y_H   = H_hist   + 0.01*rng.randn(T)
    out = Path("outputs"); out.mkdir(exist_ok=True, parents=True)
    np.savez(out/"truth.npz", ug=ug, eps_s=eps_s_hist, H=H_hist, y_eps=y_eps, y_H=y_H, dt=dt)
    # quick plots
    plt.figure(); plt.plot(eps_s_hist); plt.xlabel("t"); plt.ylabel("eps_s"); plt.title("Truth eps_s")
    plt.savefig(out/"truth_eps.png", dpi=150); plt.close()
    plt.figure(); plt.plot(H_hist); plt.xlabel("t"); plt.ylabel("H"); plt.title("Truth H")
    plt.savefig(out/"truth_H.png", dpi=150); plt.close()
    print("Saved truth to outputs/truth.npz")

if __name__=="__main__":
    simulate()
