
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt

class DragNet(nn.Module):
    def __init__(self, h=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, 1)
        )
    def forward(self, x): return self.net(x)

def main(epochs=5):
    data = np.load("outputs/truth.npz")
    ug, eps_s, H, dt = data["ug"], data["eps_s"], data["H"], float(data["dt"])

    # Use "truth" to compute the *residual* between true Cd and baseline model Cd
    def Cd_true(ug, eps): return 2.0 + 1.5*np.tanh(3*(ug-0.8)) + 0.5*np.sin(2*np.pi*eps)
    def Cd_base(ug, eps): return 2.0
    dCd = Cd_true(ug, eps_s) - Cd_base(ug, eps_s)

    x = np.stack([ug, eps_s, H], axis=1).astype(np.float32)
    y = dCd.astype(np.float32)[:,None]

    model = DragNet()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    xt = torch.from_numpy(x); yt = torch.from_numpy(y)
    losses=[]
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(xt)
        loss = loss_fn(pred, yt)
        loss.backward(); opt.step()
        losses.append(loss.item())
        print(f"Epoch {ep+1}/{epochs} loss {loss.item():.6f}")

    out = Path("outputs"); out.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), out/"dragnet.pt")

    # plot training loss
    plt.figure(); plt.plot(losses); plt.xlabel("epoch"); plt.ylabel("MSE"); plt.title("Drag residual learning")
    plt.savefig(out/"dragnet_loss.png", dpi=150); plt.close()

    # short-horizon forecast test with learned correction
    def step_model(state, ug, dCd, dt):
        eps_s, H = state
        Cd = 2.0 + dCd
        deps = dt*( -0.6*(eps_s-0.5) + 0.2*(ug-0.8) - 0.1*Cd*(eps_s-0.45) )
        dH  = dt*(  0.3*(ug-0.8) - 0.2*(eps_s-0.5) )
        eps_s = np.clip(eps_s + deps, 0.3, 0.65)
        H = np.clip(H + dH, 0.7, 1.3)
        return np.array([eps_s, H])

    state = np.array([0.5,1.0])
    horizon = 60
    states_base=[state.copy()]; states_corr=[state.copy()]
    model.eval()
    with torch.no_grad():
        for t in range(horizon):
            # baseline (dCd=0)
            s0 = step_model(states_base[-1], ug[t], 0.0, dt)
            states_base.append(s0)
            # corrected using model
            inp = torch.tensor([ug[t], states_corr[-1][0], states_corr[-1][1]], dtype=torch.float32)[None,:]
            dcd = model(inp).numpy().item()
            s1 = step_model(states_corr[-1], ug[t], dcd, dt)
            states_corr.append(s1)

    states_base=np.array(states_base); states_corr=np.array(states_corr)
    np.savez(out/"drag_corrected_forecast.npz", states_base=states_base, states_corr=states_corr)

    # plots
    plt.figure(); plt.plot(eps_s[:horizon+1],label="truth"); plt.plot(states_base[:,0],label="base"); plt.plot(states_corr[:,0],label="corrected"); plt.legend(); plt.xlabel("t"); plt.ylabel("eps_s"); plt.title("Short-horizon eps_s forecast")
    plt.savefig(out/"forecast_eps.png", dpi=150); plt.close()
    plt.figure(); plt.plot(H[:horizon+1],label="truth"); plt.plot(states_base[:,1],label="base"); plt.plot(states_corr[:,1],label="corrected"); plt.legend(); plt.xlabel("t"); plt.ylabel("H"); plt.title("Short-horizon H forecast")
    plt.savefig(out/"forecast_H.png", dpi=150); plt.close()
    print("Saved learned correction and forecasts.")

if __name__=="__main__":
    main()
