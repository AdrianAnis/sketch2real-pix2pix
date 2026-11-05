"""
train.py — Training loop, loss (BCE + 100 x L1), optimizer, metrik, checkpoint
Penanggung jawab: Meyra

Import: from src.model import Generator, Discriminator
        from src.data import ...
Jalankan dari root repo: python -m src.train
Simpan grafik loss & sampel gambar ke results/.

Kode dibangun lewat vibe coding; catat setiap sesi LLM di logs/meyra/.
"""

import argparse
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn, Tensor
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import save_image

History = Dict[str, List[float]]
EpochCallback = Callable[[int, nn.Module, nn.Module, History], None]


def train_step(
    G: nn.Module,
    D: nn.Module,
    opt_G: Optimizer,
    opt_D: Optimizer,
    sketch: Tensor,
    photo: Tensor,
    bce: nn.Module,
    l1: nn.Module,
    lambda_l1: float = 100,
) -> Tuple[float, float]:
    """Satu langkah training Pix2Pix: update D, lalu update G.

    Args:
        sketch: [B, 3, 256, 256], rentang [-1, 1].
        photo:  [B, 3, 256, 256], rentang [-1, 1].
    Returns:
        (loss_D, loss_G) sebagai float.
    """
    assert sketch.shape == photo.shape, "sketch dan photo harus sama shape"

    fake_photo = G(sketch)  # [B, 3, 256, 256]
    assert fake_photo.shape == photo.shape

    # ---------- Update D ----------
    opt_D.zero_grad()
    real_logits = D(sketch, photo)                 # [B, 1, 30, 30]
    fake_logits = D(sketch, fake_photo.detach())   # [B, 1, 30, 30], putus dari G
    loss_D_real = bce(real_logits, torch.ones_like(real_logits))
    loss_D_fake = bce(fake_logits, torch.zeros_like(fake_logits))
    loss_D = 0.5 * (loss_D_real + loss_D_fake)
    loss_D.backward()
    opt_D.step()

    # ---------- Update G ----------
    opt_G.zero_grad()
    fake_logits_for_G = D(sketch, fake_photo)      # tanpa detach: gradien ke G
    loss_G_adv = bce(fake_logits_for_G, torch.ones_like(fake_logits_for_G))
    loss_G_l1 = l1(fake_photo, photo)
    loss_G = loss_G_adv + lambda_l1 * loss_G_l1
    loss_G.backward()
    opt_G.step()

    return loss_D.item(), loss_G.item()


def pick_device() -> torch.device:
    """CUDA jika tersedia, selain itu CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _new_history() -> History:
    return {"loss_D": [], "loss_G": [], "val_l1": []}


def save_checkpoint(
    path: Union[str, Path],
    epoch: int,
    G: nn.Module,
    D: nn.Module,
    opt_G: Optimizer,
    opt_D: Optimizer,
    history: History,
) -> None:
    """Simpan checkpoint secara atomik. `epoch` = jumlah epoch yang sudah selesai."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "G": G.state_dict(),
        "D": D.state_dict(),
        "opt_G": opt_G.state_dict(),
        "opt_D": opt_D.state_dict(),
        "history": history,
    }
    tmp = Path(str(path) + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomik: file lama tidak rusak jika proses terhenti


def load_checkpoint(
    path: Union[str, Path],
    G: nn.Module,
    D: nn.Module,
    opt_G: Optimizer,
    opt_D: Optimizer,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[int, History]:
    """Pulihkan bobot dan state optimizer. Mengembalikan (epoch_selesai, history)."""
    device = torch.device(device) if device is not None else pick_device()
    state = torch.load(path, map_location=device)
    G.load_state_dict(state["G"])
    D.load_state_dict(state["D"])
    opt_G.load_state_dict(state["opt_G"])
    opt_D.load_state_dict(state["opt_D"])
    return int(state["epoch"]), state["history"]


@torch.no_grad()
def evaluate_l1(
    G: nn.Module,
    loader: DataLoader,
    device: torch.device,
    l1: nn.Module,
) -> float:
    """Rata-rata L1 antara G(sketch) dan photo di loader (dibobot ukuran batch).

    G sengaja dijalankan dalam mode train() (BatchNorm memakai statistik batch,
    Dropout aktif), sama seperti saat training dan pengujian di paper Pix2Pix.
    Tidak ada gradien (no_grad). Mode semula G dipulihkan di akhir.
    """
    was_training = G.training
    G.train()
    total, count = 0.0, 0
    try:
        for sketch, photo in loader:
            sketch, photo = sketch.to(device), photo.to(device)
            fake = G(sketch)  # [B, 3, 256, 256]
            assert fake.shape == photo.shape
            bs = sketch.size(0)
            total += l1(fake, photo).item() * bs
            count += bs
    finally:
        G.train(was_training)
    assert count > 0, "val_loader kosong"
    return total / count


def train(
    G: nn.Module,
    D: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    ckpt_dir: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    lr: float = 2e-4,
    betas: Tuple[float, float] = (0.5, 0.999),
    lambda_l1: float = 100.0,
    resume_from: Optional[Union[str, Path]] = None,
    keep_all: bool = False,
    on_epoch_end: Optional[EpochCallback] = None,
) -> History:
    """Loop training Pix2Pix.

    `G` dan `D` harus sudah di-init (init_weights) oleh pemanggil.
    `num_epochs` adalah target total epoch; jika resume, lanjut dari epoch tersimpan.
    Checkpoint selalu ditulis ke `latest.pt`; `keep_all=True` menambah `epoch_XXX.pt`.
    `on_epoch_end(epoch, G, D, history)` dipanggil setelah checkpoint tiap epoch
    (epoch berbasis 1).
    Mengembalikan history: loss_D, loss_G (rata-rata training), val_l1 per epoch.
    """
    device = torch.device(device) if device is not None else pick_device()
    ckpt_dir = Path(ckpt_dir)

    G.to(device)
    D.to(device)
    opt_G = Adam(G.parameters(), lr=lr, betas=betas)
    opt_D = Adam(D.parameters(), lr=lr, betas=betas)
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    history = _new_history()
    start_epoch = 0
    if resume_from is not None:
        start_epoch, history = load_checkpoint(resume_from, G, D, opt_G, opt_D, device)

    for epoch in range(start_epoch, num_epochs):
        G.train()
        D.train()
        sum_d, sum_g, count = 0.0, 0.0, 0
        for sketch, photo in train_loader:
            sketch = sketch.to(device)  # [B, 3, 256, 256]
            photo = photo.to(device)    # [B, 3, 256, 256]
            loss_d, loss_g = train_step(
                G, D, opt_G, opt_D, sketch, photo, bce, l1, lambda_l1
            )
            bs = sketch.size(0)
            sum_d += loss_d * bs
            sum_g += loss_g * bs
            count += bs
        assert count > 0, "train_loader kosong"

        val_l1 = evaluate_l1(G, val_loader, device, l1)
        history["loss_D"].append(sum_d / count)
        history["loss_G"].append(sum_g / count)
        history["val_l1"].append(val_l1)

        done = epoch + 1
        save_checkpoint(ckpt_dir / "latest.pt", done, G, D, opt_G, opt_D, history)
        if keep_all:
            save_checkpoint(
                ckpt_dir / f"epoch_{done:03d}.pt", done, G, D, opt_G, opt_D, history
            )
        print(
            f"[Epoch {done}/{num_epochs}] "
            f"loss_D={history['loss_D'][-1]:.4f} "
            f"loss_G={history['loss_G'][-1]:.4f} "
            f"val_L1={val_l1:.4f}"
        )

        if on_epoch_end is not None:
            on_epoch_end(done, G, D, history)

    return history


def make_sample_saver(
    sketch: Tensor,
    photo: Tensor,
    out_dir: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    num_samples: int = 8,
) -> EpochCallback:
    """Buat hook yang menyimpan grid [sketsa | hasil G | target] dari batch tetap.

    sketch, photo: [B, 3, H, W] di rentang [-1, 1]. Disimpan sebagai
    out_dir/epoch_XXX.png, satu baris per sampel.
    """
    device = torch.device(device) if device is not None else pick_device()
    sketch = sketch[:num_samples].detach().cpu()  # [N, 3, H, W]
    photo = photo[:num_samples].detach().cpu()    # [N, 3, H, W]
    out_dir = Path(out_dir)

    def save_samples(epoch: int, G: nn.Module, D: nn.Module, history: History) -> None:
        was_training = G.training
        G.train()
        try:
            with torch.no_grad():
                fake = G(sketch.to(device)).cpu()  # [N, 3, H, W]
        finally:
            G.train(was_training)
        assert fake.shape == photo.shape
        row = torch.cat([sketch, fake, photo], dim=3)  # [N, 3, H, 3W]
        row = (row.clamp(-1, 1) + 1) / 2               # [-1,1] -> [0,1]
        out_dir.mkdir(parents=True, exist_ok=True)
        save_image(row, out_dir / f"epoch_{epoch:03d}.png", nrow=1, padding=2)

    return save_samples


def save_curves(history: History, path: Union[str, Path]) -> None:
    """Simpan grafik: kiri loss D dan G, kanan val L1 (per epoch)."""
    import matplotlib
    matplotlib.use("Agg")  # tanpa layar, aman di Kaggle
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["loss_D"]) + 1)

    fig, (ax_loss, ax_val) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(epochs, history["loss_D"], label="loss D")
    ax_loss.plot(epochs, history["loss_G"], label="loss G (adv + 100xL1)")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.legend()
    ax_val.plot(epochs, history["val_l1"], color="tab:green")
    ax_val.set_xlabel("epoch")
    ax_val.set_ylabel("val L1")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Training Pix2Pix (edges2shoes)")
    parser.add_argument("--data_root", default="/kaggle/input/edges2shoes-dataset")
    parser.add_argument("--epochs", type=int, default=20, help="target total epoch")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--resume", default=None, help="path checkpoint, mis. checkpoints/latest.pt")
    parser.add_argument("--keep_all", action="store_true", help="simpan epoch_XXX.pt tiap epoch")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from src.model import Generator, Discriminator, init_weights
    from src.data import build_dataloaders

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"Device: {device}")

    train_loader, val_loader = build_dataloaders(
        args.data_root, args.batch_size, args.num_workers
    )
    fixed_sketch, fixed_photo = next(iter(val_loader))  # batch tetap, diambil sekali

    G = Generator(in_channels=3, out_channels=3, base=64)
    D = Discriminator(in_channels=6, base_ch=64)
    G.apply(init_weights)
    D.apply(init_weights)

    results_dir = Path(args.results_dir)
    save_samples = make_sample_saver(
        fixed_sketch, fixed_photo, results_dir / "samples", device, args.num_samples
    )

    def on_epoch_end(epoch: int, G: nn.Module, D: nn.Module, history: History) -> None:
        save_samples(epoch, G, D, history)
        save_curves(history, results_dir / "loss_curves.png")

    train(
        G, D, train_loader, val_loader,
        num_epochs=args.epochs,
        ckpt_dir=args.ckpt_dir,
        device=device,
        resume_from=args.resume,
        keep_all=args.keep_all,
        on_epoch_end=on_epoch_end,
    )


if __name__ == "__main__":
    main()