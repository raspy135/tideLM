"""Reference data for the Rust vision port (tests/pillow/, crates/tidelm-engine/examples/vision_check.rs),
from the checkpoint's own code in tidelm:local - run as
  docker run --rm --gpus all -v MODEL:/m:ro -v REPO:/repo:ro -v OUT:/out tidelm:local python3 /out/gen_reference.py
(this file copied into OUT):
image_processor.load_image (Pillow) patches for a set of images, and the ViT + aligner rows
(engine/vision.py's loader, on the GPU) for two of them."""
import hashlib, io, json, sys, time
sys.path[:0] = ['/repo', '/m/inference', '/m/encoding']
import numpy as np, torch
from PIL import Image, ImageDraw
import image_processor as IP

class A: pass
for k, v in json.load(open('/m/inference/config.json')).items(): setattr(A, k, v)
A.vision_enabled = True

rng = np.random.default_rng(7)
def synth(w, h):
    y, x = np.mgrid[0:h, 0:w]
    a = np.stack([(x * 255 // max(w - 1, 1)), (y * 255 // max(h - 1, 1)), ((x + y) * 7) % 256], -1).astype(np.uint8)
    im = Image.fromarray(a, 'RGB')
    d = ImageDraw.Draw(im)
    d.rectangle([w // 8, h // 6, w // 2, h // 2], fill=(200, 30, 40))
    d.ellipse([w // 2, h // 3, w - 20, h - 20], fill=(20, 160, 220))
    d.text((20, 20), "TIDE 42 vision", fill=(255, 255, 255))
    return im

def enc(im, fmt, **kw):
    b = io.BytesIO(); im.save(b, fmt, **kw); return b.getvalue()

s = synth(640, 480)
noise = Image.fromarray(rng.integers(0, 256, (333, 517, 3), dtype=np.uint8), 'RGB')
rgba = synth(64, 48).convert('RGBA'); al = np.array(rgba); al[..., 3] = np.linspace(0, 255, 64, dtype=np.uint8)[None, :]; rgba = Image.fromarray(al, 'RGBA')
g16 = Image.fromarray((np.arange(120 * 80).reshape(80, 120) * 3 % 2000).astype(np.uint16))
frames = [synth(90, 70), synth(90, 70).rotate(90)]
gb = io.BytesIO(); frames[0].save(gb, 'GIF', save_all=True, append_images=frames[1:]); gif = gb.getvalue()
images = {
    'synth.png': enc(s, 'PNG'),
    'synth.jpg': enc(s, 'JPEG', quality=88),
    'noise444.jpg': enc(noise, 'JPEG', quality=92, subsampling=0),
    'noise422.jpg': enc(noise, 'JPEG', quality=75, subsampling=1),
    'gray.jpg': enc(s.convert('L'), 'JPEG', quality=90),
    'big.jpg': enc(synth(1920, 1080), 'JPEG', quality=85),
    'small_rgba.png': enc(rgba, 'PNG'),
    'pal.png': enc(synth(300, 200).convert('P', palette=Image.ADAPTIVE, colors=50), 'PNG'),
    'tall_gray.png': enc(synth(40, 900).convert('L'), 'PNG'),
    'gray16.png': enc(g16, 'PNG'),
    'anim.gif': gif,
    'synth.webp': enc(s, 'WEBP', quality=80),
}
meta = []
for name, data in images.items():
    open(f'/out/{name}', 'wb').write(data)
    patches, vh, vw, lh, lw = IP.load_image({'data': data}, A)
    raw = patches.contiguous().view(torch.int16).numpy().tobytes()
    open(f'/out/{name}.patches', 'wb').write(raw)
    meta.append({'name': name, 'vit_h': vh, 'vit_w': vw, 'llm_h': lh, 'llm_w': lw, 'sha256': hashlib.sha256(raw).hexdigest()})
    print(name, Image.open(io.BytesIO(data)).mode, Image.open(io.BytesIO(data)).size, '->', vh, vw, lh, lw)

from engine.vision import load_vision, embed_image
t = time.time(); vit, aligner, _ = load_vision('/m', 'cuda'); print('tower load %.1f s' % (time.time() - t))
for name in ['synth.png', 'big.jpg']:
    m = next(x for x in meta if x['name'] == name)
    p = torch.frombuffer(bytearray(open(f'/out/{name}.patches', 'rb').read()), dtype=torch.int16).view(torch.bfloat16)
    p = p.view(m['vit_h'] * m['vit_w'], 3, 14, 14).cuda()
    embed_image(vit, aligner, p, m['vit_h'], m['vit_w']); torch.cuda.synchronize()
    t = time.time(); rows = embed_image(vit, aligner, p, m['vit_h'], m['vit_w']); torch.cuda.synchronize()
    m['tower_s'] = time.time() - t
    open(f'/out/{name}.rows', 'wb').write(rows.contiguous().view(torch.int16).cpu().numpy().tobytes())
    print(name, 'rows', tuple(rows.shape), 'tower %.3f s' % m['tower_s'])
json.dump(meta, open('/out/meta.json', 'w'), indent=1)
