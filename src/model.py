"""
model.py — Generator (U-Net) & Discriminator (PatchGAN 70x70)
Penanggung jawab: Rian

Kontrak:
  Generator(sketch) -> fake_photo          [B, 3, 256, 256] -> [B, 3, 256, 256]
  Discriminator(sketch, photo) -> logits   input digabung jadi 6 channel

Kode dibangun lewat vibe coding; catat setiap sesi LLM di logs/adrian/.
"""