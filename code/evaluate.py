"""FID / Inception Score evaluation, following
https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/"""
import argparse, json
import torch
import torchvision.transforms as T
from PIL import Image
from ignite.engine import Engine
from ignite.metrics import FID, InceptionScore
from ignite.metrics.gan.utils import InceptionModel
from train_gan import Generator

def interpolate(batch):  # resize to Inception-v3 input size 299x299 (as in the ignite blog)
    arr = []
    for img in batch:
        pil = T.ToPILImage()(img.cpu())
        arr.append(T.ToTensor()(pil.resize((299, 299), Image.BILINEAR)))
    return torch.stack(arr)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/wikiart64.pt")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--bs", type=int, default=100)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.manual_seed(123)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev)
    nz = ck["args"]["nz"]
    G = Generator(nz, ck["args"].get("ngf", 64)).to(dev); G.load_state_dict(ck["G"]); G.eval()

    data = torch.load(a.data)
    idx = torch.randperm(data.size(0))[: a.n]
    real_loader = torch.utils.data.DataLoader(data[idx], batch_size=a.bs)

    def evaluation_step(engine, batch):
        with torch.no_grad():
            z = torch.randn(batch.size(0), nz, device=dev)
            fake = (G(z) + 1) / 2  # [-1,1] -> [0,1]
            real = batch.float() / 255.0
            return interpolate(fake.clamp(0, 1)).to(dev), interpolate(real).to(dev)

    evaluator = Engine(evaluation_step)
    # ignite default: FID on the 1000-d softmax class probabilities (as in the blog) -> values << 1
    FID(device=dev).attach(evaluator, "fid")
    # standard FID on 2048-d pool3 features -> same scale as published FIDs
    FID(num_features=2048, feature_extractor=InceptionModel(return_features=True, device=dev),
        device=dev).attach(evaluator, "fid_pool3")
    InceptionScore(device=dev, output_transform=lambda x: x[0]).attach(evaluator, "is")
    # Inception score of the real images as a reference point
    InceptionScore(device=dev, output_transform=lambda x: x[1]).attach(evaluator, "is_real")
    evaluator.run(real_loader, max_epochs=1)
    m = {k: float(v) for k, v in evaluator.state.metrics.items()}
    m.update(ckpt=a.ckpt, n=a.n)
    print(json.dumps(m))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(m, f)
