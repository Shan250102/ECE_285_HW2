"""Loss curves, FID-vs-epoch plot and sample grids for the report."""
import argparse, glob, json, os
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from train_gan import Generator

def smooth(x, k=25):
    x = np.asarray(x)
    return np.convolve(x, np.ones(k) / k, mode="valid")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/dcgan")
    ap.add_argument("--data", default="data/wikiart64.pt")
    ap.add_argument("--out", default="report/figs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    h = json.load(open(f"{a.run}/history.json"))
    it = np.array(h["iter"])
    ck = torch.load(f"{a.run}/ckpt.pt", map_location="cpu")
    ep = (it + 1) / (len(it) / ck["epoch"])  # iteration -> fractional epoch

    evals = sorted(glob.glob(f"{a.run}/eval_ep*.json"))
    ncol = 3 if evals else 2
    fig, ax = plt.subplots(1, ncol, figsize=(4.2 * ncol, 2.6))
    k = max(25, len(it) // 300)  # moving-average window scales with run length
    ax[0].plot(ep, h["lossD"], color="C0", alpha=0.15, lw=0.6)
    ax[0].plot(ep, h["lossG"], color="C1", alpha=0.15, lw=0.6)
    ax[0].plot(ep[k - 1:], smooth(h["lossD"], k), color="C0", label="Discriminator")
    ax[0].plot(ep[k - 1:], smooth(h["lossG"], k), color="C1", label="Generator")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("BCE loss"); ax[0].legend(frameon=False); ax[0].set_title("Training losses")
    ax[1].plot(ep[k - 1:], smooth(h["D_x"], k), color="C2", label=r"$D(x)$ real")
    ax[1].plot(ep[k - 1:], smooth(h["D_G_z"], k), color="C3", label=r"$D(G(z))$ fake")
    ax[1].set_xlabel("epoch"); ax[1].set_ylim(0, 1); ax[1].legend(frameon=False); ax[1].set_title("Discriminator outputs")
    best = f"{a.run}/ckpt.pt"
    if evals:
        ms = [json.load(open(f)) for f in evals]
        e = [int(os.path.basename(f)[7:10]) for f in evals]
        key = "fid_pool3" if all("fid_pool3" in m for m in ms) else "fid"  # prefer standard FID
        ax[2].plot(e, [m[key] for m in ms], "o-", color="C4", label="FID")
        ax[2].set_xlabel("epoch"); ax[2].set_ylabel("FID (pool3)" if key == "fid_pool3" else "FID", color="C4")
        ax[2].set_title("FID / IS vs. epoch")
        bm = min(ms, key=lambda m: m[key])
        best = bm["ckpt"]
        print(f"best checkpoint by {key}: {best} ({key}={bm[key]:.4f}, IS={bm['is']:.3f})")
        t = ax[2].twinx()
        t.plot(e, [m["is"] for m in ms], "s--", color="C5", label="IS"); t.set_ylabel("IS", color="C5")
    for x in ax: x.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{a.out}/curves.pdf"); plt.savefig(f"{a.out}/curves.png", dpi=150)

    # generated samples (best checkpoint by FID) and real samples for comparison
    G = Generator(ck["args"]["nz"], ck["args"]["ngf"]); G.eval()
    G.load_state_dict(torch.load(best, map_location="cpu")["G"])
    torch.manual_seed(7)
    with torch.no_grad():
        fake = G(torch.randn(48, ck["args"]["nz"]))
    vutils.save_image(fake, f"{a.out}/fake.png", nrow=12, normalize=True, value_range=(-1, 1), padding=1)
    real = torch.load(a.data)
    real = real[torch.randperm(real.size(0))[:24]].float() / 255
    vutils.save_image(real, f"{a.out}/real.png", nrow=12, padding=1)
    # progression: same fixed noise at several epochs
    prog = sorted(glob.glob(f"{a.run}/ckpt_ep*.pt"))
    if len(prog) > 5:  # keep at most 5 evenly spaced rows so the figure fits the report
        prog = [prog[i] for i in np.linspace(0, len(prog) - 1, 5).round().astype(int)]
    print("progress rows:", [os.path.basename(p) for p in prog])
    if prog:
        z = torch.randn(10, ck["args"]["nz"])
        rows = []
        for p in prog:
            G.load_state_dict(torch.load(p, map_location="cpu")["G"])
            with torch.no_grad():
                rows.append(G(z))
        vutils.save_image(torch.cat(rows), f"{a.out}/progress.png", nrow=10, normalize=True, value_range=(-1, 1), padding=1)
    print("figures written to", a.out)
