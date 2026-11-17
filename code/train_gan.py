"""DCGAN (Radford et al., 2016) trained on 64x64 WikiArt paintings."""
import argparse, json, os, time
import torch, torch.nn as nn
import torchvision.utils as vutils

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64):
        super().__init__()
        def up(i, o):
            return [nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True)]
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True),  # 4x4
            *up(ngf * 8, ngf * 4),  # 8x8
            *up(ngf * 4, ngf * 2),  # 16x16
            *up(ngf * 2, ngf),      # 32x32
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False), nn.Tanh())  # 64x64

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))

class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        def down(i, o):
            return [nn.Conv2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, True)]
        self.net = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),  # 32
            *down(ndf, ndf * 2),      # 16
            *down(ndf * 2, ndf * 4),  # 8
            *down(ndf * 4, ndf * 8),  # 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False))  # logit

    def forward(self, x):
        return self.net(x).view(-1)

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/wikiart64.pt")
    ap.add_argument("--out", default="runs/dcgan")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--nz", type=int, default=100)
    ap.add_argument("--ngf", type=int, default=32)
    ap.add_argument("--ndf", type=int, default=32)
    ap.add_argument("--smooth", type=float, default=0.9, help="one-sided label smoothing for real labels")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = torch.load(a.data)  # uint8 [N,3,64,64]
    N = data.size(0)
    print(f"{N} training images, device={dev}", flush=True)

    G, D = Generator(a.nz, a.ngf).to(dev), Discriminator(a.ndf).to(dev)
    G.apply(weights_init); D.apply(weights_init)
    optG = torch.optim.Adam(G.parameters(), lr=a.lr, betas=(a.beta1, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=a.lr, betas=(a.beta1, 0.999))
    bce = nn.BCEWithLogitsLoss()
    fixed_z = torch.randn(64, a.nz, device=dev)
    hist = {"iter": [], "lossD": [], "lossG": [], "D_x": [], "D_G_z": []}
    it, t0 = 0, time.time()

    for ep in range(a.epochs):
        perm = torch.randperm(N)
        for i in range(0, N - a.bs + 1, a.bs):
            real = data[perm[i:i + a.bs]].to(dev).float().div(127.5).sub(1)
            flip = torch.rand(real.size(0), device=dev) < 0.5  # random horizontal flip
            real[flip] = real[flip].flip(-1)
            b = real.size(0)
            z = torch.randn(b, a.nz, device=dev)
            fake = G(z)

            # D step: maximize log D(x) + log(1 - D(G(z)))
            optD.zero_grad()
            out_r, out_f = D(real), D(fake.detach())
            lossD = bce(out_r, torch.full((b,), a.smooth, device=dev)) + bce(out_f, torch.zeros(b, device=dev))
            lossD.backward(); optD.step()

            # G step: non-saturating loss, maximize log D(G(z))
            optG.zero_grad()
            out_g = D(fake)
            lossG = bce(out_g, torch.ones(b, device=dev))
            lossG.backward(); optG.step()

            hist["iter"].append(it); hist["lossD"].append(lossD.item()); hist["lossG"].append(lossG.item())
            hist["D_x"].append(torch.sigmoid(out_r).mean().item()); hist["D_G_z"].append(torch.sigmoid(out_f).mean().item())
            it += 1
        print(f"epoch {ep + 1}/{a.epochs} it {it} lossD {lossD.item():.3f} lossG {lossG.item():.3f} "
              f"D(x) {hist['D_x'][-1]:.2f} D(G(z)) {hist['D_G_z'][-1]:.2f} [{time.time() - t0:.0f}s]", flush=True)
        G.eval()
        with torch.no_grad():
            vutils.save_image(G(fixed_z), f"{a.out}/samples_ep{ep + 1:03d}.png", nrow=8, normalize=True, value_range=(-1, 1))
        G.train()
        ck = {"G": G.state_dict(), "D": D.state_dict(), "args": vars(a), "epoch": ep + 1}
        torch.save(ck, f"{a.out}/ckpt.pt")
        if (ep + 1) % 10 == 0:
            torch.save(ck, f"{a.out}/ckpt_ep{ep + 1:03d}.pt")
        with open(f"{a.out}/history.json", "w") as f:
            json.dump(hist, f)
    print("done", time.time() - t0)
