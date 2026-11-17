# ECE 285 HW2 — DCGAN on WikiArt

A DCGAN trained on 64×64 WikiArt paintings, evaluated with Inception Score and FID
(following the [PyTorch-Ignite GAN evaluation guide](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/)).

## Layout
```
code/
  prepare_data.py   # build a uint8 [N,3,64,64] WikiArt cache (Kaggle dir or HF parquet shards)
  train_gan.py      # DCGAN training
  evaluate.py       # FID / IS of generated samples
  make_figures.py   # loss curves, FID-vs-epoch, sample grids
report/
  report.tex, neurips_2025.sty, report.pdf
  figs_full/        # figures used in the report
```

## Reproduce
```bash
python code/prepare_data.py --out data/wikiart64.pt            # or --kaggle_dir /path/to/wikiart
python code/train_gan.py --data data/wikiart64.pt --out runs/dcgan
python code/evaluate.py --ckpt runs/dcgan/<checkpoint> --data data/wikiart64.pt
python code/make_figures.py --run runs/dcgan --out report/figs_full
```

The dataset cache (`data/`) and checkpoints (`runs/`) are not committed because of their size.
