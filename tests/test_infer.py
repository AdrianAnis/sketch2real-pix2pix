from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw, ImageOps
from torch import nn

from src.infer import (
    IMG_SIZE,
    compose_side_by_side,
    list_val_images,
    load_context_sketches,
    load_generator,
    main,
    preprocess_sketch,
    sketch_to_photo,
)
from src.model import Generator


# ----------------------------------------------------------------- helper
def make_sketch(kind: int = 0, size: int = IMG_SIZE) -> Image.Image:
    """Sketsa sintetis: garis hitam di latar putih."""
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    if kind == 0:
        d.ellipse([40, 90, 220, 170], outline="black", width=4)
        d.line([60, 130, 200, 130], fill="black", width=3)
    elif kind == 1:
        d.rectangle([60, 60, 200, 200], outline="black", width=5)
        d.line([60, 60, 200, 200], fill="black", width=3)
    else:
        d.line([(30, 200), (120, 60), (230, 200), (30, 200)], fill="black", width=4)
    return img


def lum(img: Image.Image, xy: tuple[int, int]) -> int:
    return img.convert("L").getpixel(xy)


def as_arr(img: Image.Image) -> np.ndarray:
    return np.asarray(img).astype(np.int16)


class _ConstG(nn.Module):
    """Generator pengganti: keluaran konstan (untuk tes clamp/pemetaan)."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, self.value) + 0 * self.w


class _IdentityG(nn.Module):
    """Generator pengganti: G(x) = x (untuk tes indeks 0 dan pemetaan)."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0 * self.w


CONTEXT_A = [make_sketch(1), make_sketch(2)]
CONTEXT_B = [make_sketch(2), make_sketch(0)]


# --------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def ckpt_path(tmp_path_factory):
    torch.manual_seed(0)
    G = Generator(in_channels=3, out_channels=3, base=64)
    path = tmp_path_factory.mktemp("ckpt") / "latest.pt"
    torch.save(
        {
            "epoch": 1,
            "G": G.state_dict(),
            "D": {"w": torch.zeros(1)},
            "history": {"l1": [0.5, 0.4]},
        },
        path,
    )
    return path


@pytest.fixture(scope="module")
def G(ckpt_path):
    return load_generator(ckpt_path, "cpu")


def make_val_dir(tmp_path, n: int = 2):
    d = tmp_path / "val"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n):
        left = make_sketch(i)
        right = Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))
        pair = Image.new("RGB", (512, 256))
        pair.paste(left, (0, 0))
        pair.paste(right, (256, 0))
        pair.save(d / f"{i}.png")
    return d


# ------------------------------------------------------- load_generator
def test_load_generator_full_checkpoint(ckpt_path):
    G = load_generator(ckpt_path, "cpu")
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=True)["G"]
    assert G.training is True
    n_bn = 0
    for m in G.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            n_bn += 1
            assert m.momentum == 0.0
    assert n_bn > 0
    for k, v in G.state_dict().items():
        assert torch.equal(v, saved[k]), k


def test_load_generator_state_dict_only(ckpt_path, tmp_path):
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=True)["G"]
    path = tmp_path / "g_only.pt"
    torch.save(saved, path)
    G = load_generator(path, "cpu")
    for k, v in G.state_dict().items():
        assert torch.equal(v, saved[k]), k


def test_load_generator_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_generator(tmp_path / "tidak_ada.pt", "cpu")
    empty = tmp_path / "empty.pt"
    torch.save({}, empty)
    with pytest.raises(ValueError):
        load_generator(empty, "cpu")
    wrong = tmp_path / "wrong.pt"
    torch.save({"foo": torch.zeros(1)}, wrong)
    with pytest.raises(RuntimeError):  # arsitektur/kunci tidak cocok (strict)
        load_generator(wrong, "cpu")


# ------------------------------------------------------- preprocess_sketch
def test_preprocess_light_background_unchanged():
    src = make_sketch(0)
    out = preprocess_sketch(src)
    assert out.size == (IMG_SIZE, IMG_SIZE) and out.mode == "RGB"
    assert np.array_equal(as_arr(out), as_arr(src))


def test_preprocess_auto_inverts_dark_background():
    dark = ImageOps.invert(make_sketch(0))
    out = preprocess_sketch(dark)
    assert lum(out, (0, 0)) == 255
    assert np.array_equal(as_arr(out), as_arr(make_sketch(0)))


def test_preprocess_invert_override():
    black = Image.new("RGB", (IMG_SIZE, IMG_SIZE), "black")
    white = Image.new("RGB", (IMG_SIZE, IMG_SIZE), "white")
    assert lum(preprocess_sketch(black, invert=False), (0, 0)) == 0
    assert lum(preprocess_sketch(white, invert=True), (0, 0)) == 0


@pytest.mark.parametrize("size", [(300, 200), (200, 300), (512, 512), (64, 64), (1000, 10)])
def test_preprocess_output_size(size):
    out = preprocess_sketch(Image.new("RGB", size, "white"))
    assert out.size == (IMG_SIZE, IMG_SIZE)


def test_preprocess_letterbox_pads_white_and_keeps_ratio():
    img = Image.new("RGB", (400, 100), "white")
    ImageDraw.Draw(img).rectangle([100, 0, 110, 99], fill="black")
    out = preprocess_sketch(img)
    # skala 0,64 -> konten tinggi 64, offset y = 96; padding atas/bawah putih
    assert lum(out, (0, 0)) == 255
    assert lum(out, (0, 95)) == 255
    assert lum(out, (0, 255)) == 255
    assert lum(out, (67, 128)) < 60  # garis ikut diskalakan ke x ~ 67


def test_preprocess_transparent_png_composited_on_white():
    img = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), (0, 0, 0, 0))
    ImageDraw.Draw(img).rectangle([100, 120, 150, 130], fill=(0, 0, 0, 255))
    out = preprocess_sketch(img)
    assert lum(out, (0, 0)) == 255  # transparan -> putih, bukan hitam
    assert lum(out, (125, 125)) < 50


@pytest.mark.parametrize("mode", ["L", "1", "RGBA"])
def test_preprocess_other_modes(mode):
    out = preprocess_sketch(Image.new(mode, (50, 40), 1 if mode == "1" else 255))
    assert out.mode == "RGB" and out.size == (IMG_SIZE, IMG_SIZE)


# ------------------------------------------- compose / list / context loader
def test_compose_side_by_side():
    panels = [Image.new("RGB", (IMG_SIZE, IMG_SIZE), c) for c in ("red", "lime", "blue")]
    out = compose_side_by_side(panels)
    assert out.size == (IMG_SIZE * 3, IMG_SIZE)
    assert out.getpixel((10, 10)) == (255, 0, 0)
    assert out.getpixel((IMG_SIZE + 10, 10)) == (0, 255, 0)
    assert out.getpixel((2 * IMG_SIZE + 10, 10)) == (0, 0, 255)


def test_compose_rejects_bad_input():
    with pytest.raises(ValueError):
        compose_side_by_side([])
    with pytest.raises(ValueError):
        compose_side_by_side([Image.new("RGB", (100, 100))])


def test_list_val_images_sorted_and_filtered(tmp_path):
    for name in ("b.jpg", "a.png", "c.JPG", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    assert [p.name for p in list_val_images(tmp_path)] == ["a.png", "b.jpg", "c.JPG"]
    with pytest.raises(FileNotFoundError):
        list_val_images(tmp_path / "tidak_ada")


def test_load_context_missing_or_empty_dir_warns(tmp_path):
    with pytest.warns(UserWarning, match="batch 1"):
        assert load_context_sketches(tmp_path / "tidak_ada") == []
    empty = tmp_path / "kosong"
    empty.mkdir()
    with pytest.warns(UserWarning, match="batch 1"):
        assert load_context_sketches(empty) == []


def test_load_context_sorted_preprocessed_png_only(tmp_path):
    # tulis terbalik; urutan hasil harus mengikuti nama file
    for name, width in (("02.png", 80), ("00.png", 10), ("01.png", 40)):
        img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), "white")
        ImageDraw.Draw(img).rectangle([0, 0, width - 1, IMG_SIZE - 1], fill="black")
        img.save(tmp_path / name)
    Image.new("RGB", (50, 50), "white").save(tmp_path / "x.jpg")  # harus diabaikan
    # sketsa berukuran lain + latar gelap juga harus dinormalkan
    ImageOps.invert(Image.new("RGB", (300, 200), "white")).save(tmp_path / "03.png")
    out = load_context_sketches(tmp_path)
    assert len(out) == 4
    assert all(o.size == (IMG_SIZE, IMG_SIZE) and o.mode == "RGB" for o in out)
    means = [float(np.asarray(o.convert("L")).mean()) for o in out[:3]]
    assert means[0] > means[1] > means[2]


# ------------------------------------------------------- sketch_to_photo
def test_output_type_size_dtype(G):
    out = sketch_to_photo(G, make_sketch(0), seed=0)
    assert isinstance(out, Image.Image)
    assert out.size == (IMG_SIZE, IMG_SIZE) and out.mode == "RGB"
    arr = np.asarray(out)
    assert arr.dtype == np.uint8 and arr.shape == (IMG_SIZE, IMG_SIZE, 3)


def test_batch1_no_error_with_none_and_empty_context(G):
    assert sketch_to_photo(G, make_sketch(0), seed=0).size == (IMG_SIZE, IMG_SIZE)
    assert sketch_to_photo(G, make_sketch(0), seed=0, context=[]).size == (IMG_SIZE, IMG_SIZE)


def test_batch1_equals_empty_context(G):
    a = sketch_to_photo(G, make_sketch(0), seed=3)
    b = sketch_to_photo(G, make_sketch(0), seed=3, context=[])
    assert np.array_equal(as_arr(a), as_arr(b))


def test_same_seed_same_output(G):
    a = sketch_to_photo(G, make_sketch(0), seed=7)
    b = sketch_to_photo(G, make_sketch(0), seed=7)
    assert np.array_equal(as_arr(a), as_arr(b))


def test_same_seed_same_context_same_output(G):
    a = sketch_to_photo(G, make_sketch(0), seed=7, context=CONTEXT_A)
    b = sketch_to_photo(G, make_sketch(0), seed=7, context=CONTEXT_A)
    assert a.size == (IMG_SIZE, IMG_SIZE)
    assert np.array_equal(as_arr(a), as_arr(b))


def test_different_seed_different_output(G):
    a = sketch_to_photo(G, make_sketch(0), seed=1)
    b = sketch_to_photo(G, make_sketch(0), seed=2)
    assert not np.array_equal(as_arr(a), as_arr(b))  # Dropout aktif


def test_different_context_different_output(G):
    base = sketch_to_photo(G, make_sketch(0), seed=5)
    a = sketch_to_photo(G, make_sketch(0), seed=5, context=CONTEXT_A)
    b = sketch_to_photo(G, make_sketch(0), seed=5, context=CONTEXT_B)
    assert not np.array_equal(as_arr(a), as_arr(b))
    assert not np.array_equal(as_arr(a), as_arr(base))


def test_context_wrong_size_raises(G):
    with pytest.raises(ValueError):
        sketch_to_photo(G, make_sketch(0), seed=0, context=[Image.new("RGB", (100, 100), "white")])


def test_user_sketch_is_index_zero_and_mapping_roundtrip():
    sk = make_sketch(0)
    out = sketch_to_photo(_IdentityG(), sk, seed=0, context=[make_sketch(1), make_sketch(2)])
    assert np.abs(as_arr(out) - as_arr(preprocess_sketch(sk))).max() <= 1


@pytest.mark.parametrize("value, expected", [(5.0, 255), (-5.0, 0)])
def test_postprocess_clamps_out_of_range(value, expected):
    out = sketch_to_photo(_ConstG(value), make_sketch(0), seed=0)
    assert np.all(as_arr(out) == expected)


def test_postprocess_zero_maps_to_mid_gray():
    out = sketch_to_photo(_ConstG(0.0), make_sketch(0), seed=0)
    assert np.all(np.abs(as_arr(out) - 127.5) <= 0.5)


def test_dark_background_matches_inverted_input(G):
    light = make_sketch(0)
    dark = ImageOps.invert(light)
    a = sketch_to_photo(G, light, seed=0)
    b = sketch_to_photo(G, dark, seed=0)  # auto-invert
    assert np.array_equal(as_arr(a), as_arr(b))


def test_global_rng_state_restored(G):
    torch.manual_seed(123)
    before = torch.get_rng_state()
    sketch_to_photo(G, make_sketch(0), seed=5, context=CONTEXT_A)
    assert torch.equal(torch.get_rng_state(), before)


def test_forces_train_mode(G):
    G.eval()
    sketch_to_photo(G, make_sketch(0), seed=0)
    assert G.training is True


def test_bn_running_stats_unchanged_after_inference(ckpt_path):
    G2 = load_generator(ckpt_path, "cpu")
    keys = ("running_mean", "running_var")  # num_batches_tracked tetap bertambah, itu wajar
    before = {k: v.clone() for k, v in G2.state_dict().items() if k.endswith(keys)}
    assert before
    sketch_to_photo(G2, make_sketch(0), seed=0, context=CONTEXT_A)
    after = G2.state_dict()
    for k, v in before.items():
        assert torch.equal(after[k], v), k


# ------------------------------------------------------------------- CLI
def _base_args(ckpt_path, out):
    return ["--ckpt", str(ckpt_path), "--output", str(out), "--device", "cpu", "--seed", "0"]


def test_cli_input_no_context(ckpt_path, tmp_path, capsys):
    inp = tmp_path / "sketsa.png"
    make_sketch(0).save(inp)
    out = tmp_path / "sub" / "hasil.png"  # folder induk dibuat otomatis
    rc = main(_base_args(ckpt_path, out) + ["--input", str(inp), "--no_context"])
    assert rc == 0 and out.is_file()
    res = Image.open(out)
    assert res.size == (IMG_SIZE * 2, IMG_SIZE)  # sketsa | hasil
    assert np.array_equal(as_arr(res.crop((0, 0, IMG_SIZE, IMG_SIZE))), as_arr(make_sketch(0)))
    assert "batch=1" in capsys.readouterr().out


def test_cli_with_context_dir(ckpt_path, tmp_path, capsys):
    inp = tmp_path / "sketsa.png"
    make_sketch(0).save(inp)
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    make_sketch(1).save(ctx / "00.png")
    make_sketch(2).save(ctx / "01.png")
    out = tmp_path / "hasil.png"
    rc = main(_base_args(ckpt_path, out) + ["--input", str(inp), "--context_dir", str(ctx)])
    assert rc == 0
    assert "batch=3" in capsys.readouterr().out


def test_cli_same_seed_same_file(ckpt_path, tmp_path):
    inp = tmp_path / "sketsa.png"
    make_sketch(0).save(inp)
    o1, o2 = tmp_path / "a.png", tmp_path / "b.png"
    for o in (o1, o2):
        assert main(_base_args(ckpt_path, o) + ["--input", str(inp), "--no_context"]) == 0
    assert np.array_equal(as_arr(Image.open(o1)), as_arr(Image.open(o2)))


def test_cli_val_index_three_panels(ckpt_path, tmp_path):
    val = make_val_dir(tmp_path)
    out = tmp_path / "val1.png"
    rc = main(
        _base_args(ckpt_path, out)
        + ["--val_dir", str(val), "--val_index", "1", "--no_context"]
    )
    assert rc == 0
    res = Image.open(out).convert("RGB")
    assert res.size == (IMG_SIZE * 3, IMG_SIZE)  # sketsa | hasil | asli
    pair = Image.open(val / "1.png").convert("RGB")
    left = pair.crop((0, 0, 256, 256))
    right = pair.crop((256, 0, 512, 256))
    assert np.array_equal(as_arr(res.crop((0, 0, 256, 256))), as_arr(left))
    assert np.array_equal(as_arr(res.crop((512, 0, 768, 256))), as_arr(right))


@pytest.mark.parametrize("idx", ["5", "-1"])
def test_cli_val_index_out_of_range(ckpt_path, tmp_path, capsys, idx):
    val = make_val_dir(tmp_path)
    rc = main(
        _base_args(ckpt_path, tmp_path / "x.png")
        + ["--val_dir", str(val), f"--val_index={idx}", "--no_context"]
    )
    assert rc == 1
    assert capsys.readouterr().err.startswith("Error:")


def test_cli_runtime_errors_are_one_line(ckpt_path, tmp_path, capsys):
    rc = main(_base_args(ckpt_path, tmp_path / "x.png") + ["--input", str(tmp_path / "x.png")])
    assert rc == 1 and capsys.readouterr().err.startswith("Error:")  # input tidak ada

    inp = tmp_path / "sketsa.png"
    make_sketch(0).save(inp)
    rc = main(_base_args(tmp_path / "tidak_ada.pt", tmp_path / "y.png")
              + ["--input", str(inp), "--no_context"])
    assert rc == 1 and capsys.readouterr().err.startswith("Error:")  # ckpt tidak ada

    rc = main(_base_args(ckpt_path, tmp_path / "z.png")
              + ["--val_dir", str(tmp_path / "tidak_ada"), "--val_index", "0"])
    assert rc == 1 and capsys.readouterr().err.startswith("Error:")  # val_dir tidak ada


@pytest.mark.parametrize(
    "extra",
    [
        [],                                                  # tidak ada --input / --val_index
        ["--input", "a.png", "--val_index", "0", "--val_dir", "d"],  # keduanya
        ["--val_index", "0"],                                # tanpa --val_dir
    ],
)
def test_cli_usage_errors_exit_2(ckpt_path, tmp_path, extra):
    with pytest.raises(SystemExit) as exc:
        main(["--ckpt", str(ckpt_path), "--output", str(tmp_path / "o.png")] + extra)
    assert exc.value.code == 2


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA tersedia")
def test_cli_cuda_requested_but_unavailable(ckpt_path, tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--ckpt", str(ckpt_path), "--output", str(tmp_path / "o.png"),
              "--input", "a.png", "--device", "cuda"])
    assert exc.value.code == 2
