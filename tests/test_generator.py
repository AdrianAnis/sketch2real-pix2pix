import torch

from src.model import Generator


def test_generator_shape_and_range() -> None:
    torch.manual_seed(0)
    gen = Generator()
    x = torch.rand(2, 3, 256, 256) * 2 - 1  # dummy sketch dalam [-1, 1]

    skip_shapes: list[tuple[int, ...]] = []
    hooks = [
        down.register_forward_hook(lambda m, i, o: skip_shapes.append(tuple(o.shape)))
        for down in gen.downs
    ]

    out = gen(x)
    for h in hooks:
        h.remove()

    print("Skip connections (sebelum di-concat):")
    for i, s in enumerate(skip_shapes[:-1], start=1):
        print(f"  d{i}: {s}  -> di-concat dengan output up{8 - i}")
    print(f"  d8 (bottleneck, bukan skip): {skip_shapes[-1]}")
    print(f"Output: {tuple(out.shape)}, min={out.min():.4f}, max={out.max():.4f}")

    assert out.shape == (2, 3, 256, 256)
    assert out.dtype == torch.float32
    assert out.min() >= -1.0 and out.max() <= 1.0  # <= karena Tanh float32 bisa saturasi tepat di ±1
    assert skip_shapes[-1] == (2, 512, 1, 1)


if __name__ == "__main__":
    test_generator_shape_and_range()
    print("OK")