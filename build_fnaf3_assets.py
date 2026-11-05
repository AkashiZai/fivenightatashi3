import os, urllib.request
from collections import deque
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

ROOT = os.path.dirname(os.path.abspath(__file__))
BG_DIR = os.path.join(ROOT, 'assets3', 'bg')
SPRITE_DIR = os.path.join(ROOT, 'assets3', 'sprites')
CAM_DIR = os.path.join(ROOT, 'assets3', 'cams')

os.makedirs(BG_DIR, exist_ok=True)
os.makedirs(SPRITE_DIR, exist_ok=True)
os.makedirs(CAM_DIR, exist_ok=True)

# 1. Cleanly segment Springtrap Ashi from the uploaded hallway image using Polygon + Color + Flood-Fill
USER_SPRINGTRAP = r"C:\Users\pisit\.gemini\antigravity\brain\b2a35b52-a5d9-4376-9dab-fec01759a477\.user_uploaded\media_1793899363009_f6e06fc6.jpg"
st_raw = Image.open(USER_SPRINGTRAP).convert('RGBA')
w, h = st_raw.size
arr = np.array(st_raw, dtype=np.float32)
r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

# Draw a tight anatomical polygon silhouette around Springtrap Ashi (normalized coords)
poly_mask_img = Image.new('L', (w, h), 0)
draw_p = ImageDraw.Draw(poly_mask_img)

def pt(nx, ny):
    return (int(nx * w), int(ny * h))

# Main body + head + legs + arms tight contour
# Left ear:
draw_p.polygon([pt(0.28,0.18), pt(0.22,0.09), pt(0.32,0.07), pt(0.39,0.14), pt(0.38,0.19)], fill=255)
# Right ear:
draw_p.polygon([pt(0.49,0.15), pt(0.57,0.00), pt(0.74,0.00), pt(0.60,0.18)], fill=255)
# Head:
draw_p.ellipse([pt(0.28, 0.13), pt(0.61, 0.30)], fill=255)
# Neck & Shoulders & Torso:
draw_p.polygon([
    pt(0.38, 0.28), pt(0.53, 0.28),
    pt(0.75, 0.27), pt(0.87, 0.34), pt(0.92, 0.48), pt(0.91, 0.65), pt(0.74, 0.67), pt(0.72, 0.45),
    pt(0.68, 0.56), pt(0.76, 0.67), pt(0.75, 0.99), pt(0.57, 0.99), pt(0.55, 0.63),
    pt(0.45, 0.63), pt(0.43, 0.99), pt(0.26, 0.99), pt(0.29, 0.56),
    pt(0.25, 0.44), pt(0.27, 0.67), pt(0.08, 0.66), pt(0.08, 0.46), pt(0.12, 0.33), pt(0.25, 0.28)
], fill=255)

sil = np.array(poly_mask_img) > 128

# Refine inside the silhouette: remove low-saturation background hallway pixels near edges
max_c = np.maximum(np.maximum(r, g), b)
min_c = np.minimum(np.minimum(r, g), b)
sat = (max_c - min_c) / (max_c + 1.0)
green_warm = (g - b) + (r - b) * 0.5

# Hallway wall/floor is desaturated grey/beige (sat < 0.25 and green_warm < 28)
is_wall_like = (sat < 0.24) & (green_warm < 26) & (max_c > 32)
# Flood-fill wall-like pixels starting from outside the silhouette
bg_visited = ~sil.copy()
q = deque()
H_s, W_s = sil.shape
for y in range(H_s):
    for x in range(W_s):
        if bg_visited[y, x]:
            q.append((y, x))

while q:
    cy, cx = q.popleft()
    for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
        ny_i, nx_i = cy + dy, cx + dx
        if 0 <= ny_i < H_s and 0 <= nx_i < W_s and not bg_visited[ny_i, nx_i]:
            if is_wall_like[ny_i, nx_i]:
                bg_visited[ny_i, nx_i] = True
                q.append((ny_i, nx_i))

# Always keep Ashi's face inside the Springtrap head!
yy, xx = np.mgrid[0:h, 0:w]
face_oval = (((xx - w*0.44)/(w*0.13))**2 + ((yy - h*0.215)/(h*0.075))**2) <= 1.0
final_fg = ((~bg_visited) | face_oval).astype(np.uint8) * 255

mask_img = Image.fromarray(final_fg, mode='L').filter(ImageFilter.MedianFilter(5)).filter(ImageFilter.GaussianBlur(2.2))
st_cutout = st_raw.copy()
st_cutout.putalpha(mask_img)
st_cutout.save(os.path.join(SPRITE_DIR, 'springtrap.png'))
st_raw.convert('RGB').save(os.path.join(SPRITE_DIR, 'springtrap_full.jpg'), quality=92)
print("Saved clean Springtrap cutout -> assets3/sprites/springtrap.png")

# 2. Create realistic charred FNaF 3 Phantom Ashi sprites
def make_phantom(src_path, out_name, tint=(0.48, 0.72, 0.34)):
    im = Image.open(src_path).convert('RGBA')
    a = np.array(im, dtype=np.float32)
    alpha = a[:,:,3]
    rgb = a[:,:,:3]
    lum = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]

    h_i, w_i = lum.shape
    yy_i, xx_i = np.mgrid[0:h_i, 0:w_i]
    # Organic multi-scale char/burn streaks
    burn = (
        0.72
        + 0.18 * np.sin(xx_i * 0.045 + yy_i * 0.015)
        + 0.10 * np.cos(yy_i * 0.08 - xx_i * 0.02)
    )
    scorch = np.clip((lum / 255.0) ** 1.45 * 1.40, 0, 1) * burn

    out = np.zeros_like(a)
    # Blend original texture details with charred green-black FNaF 3 Phantom palette
    out[:,:,0] = np.clip((scorch * 210.0 * tint[0]) + (rgb[:,:,0] * 0.18 * tint[0]), 0, 255)
    out[:,:,1] = np.clip((scorch * 235.0 * tint[1]) + (rgb[:,:,1] * 0.22 * tint[1]), 0, 255)
    out[:,:,2] = np.clip((scorch * 190.0 * tint[2]) + (rgb[:,:,2] * 0.15 * tint[2]), 0, 255)
    out[:,:,3] = alpha

    res = Image.fromarray(out.astype(np.uint8), mode='RGBA')
    res = ImageEnhance.Contrast(res).enhance(1.38)
    res.save(os.path.join(SPRITE_DIR, out_name))
    print("  Created Phantom sprite:", out_name)

make_phantom(os.path.join(ROOT, 'assets', 'sprites', 'freddy.png'),  'phantom_freddy.png', tint=(0.45, 0.72, 0.32))
make_phantom(os.path.join(ROOT, 'assets', 'sprites', 'chica.png'),   'phantom_chica.png',  tint=(0.48, 0.75, 0.30))
make_phantom(os.path.join(ROOT, 'assets', 'sprites', 'foxy.png'),    'phantom_foxy.png',   tint=(0.54, 0.68, 0.28))
make_phantom(os.path.join(ROOT, 'assets2', 'sprites', 'bonnie.png'), 'phantom_bb.png',     tint=(0.46, 0.72, 0.36))
make_phantom(os.path.join(ROOT, 'assets', 'sprites', 'bonnie.png'),  'phantom_mangle.png', tint=(0.42, 0.74, 0.33))
make_phantom(os.path.join(ROOT, 'assets2', 'sprites', 'puppet.png'), 'phantom_puppet.png', tint=(0.40, 0.76, 0.35))

# 3. Composite all 47 FNaF 3 Office & Camera frames
def style_for_fnaf3(sp, brightness=0.68, contrast=1.35, green_shift=True):
    sp = ImageEnhance.Brightness(sp).enhance(brightness)
    sp = ImageEnhance.Contrast(sp).enhance(contrast)
    if green_shift:
        a = np.array(sp, dtype=np.float32)
        a[:,:,0] = np.clip(a[:,:,0] * 0.85, 0, 255)
        a[:,:,1] = np.clip(a[:,:,1] * 1.08, 0, 255)
        a[:,:,2] = np.clip(a[:,:,2] * 0.72, 0, 255)
        sp = Image.fromarray(a.astype(np.uint8), mode='RGBA')
    return sp

def paste_char(bg, sprite_name, scale_h, center_x, bottom_y, brightness=0.68, contrast=1.35, alpha_mul=1.0, crop_bottom=1.0):
    canvas = bg.copy().convert('RGBA')
    sp = Image.open(os.path.join(SPRITE_DIR, sprite_name)).convert('RGBA')
    if crop_bottom < 1.0:
        sp = sp.crop((0, 0, sp.width, int(sp.height * crop_bottom)))
    sp = style_for_fnaf3(sp, brightness=brightness, contrast=contrast)
    ratio = scale_h / float(sp.height)
    new_w = max(1, int(sp.width * ratio))
    sp = sp.resize((new_w, int(scale_h)), Image.Resampling.LANCZOS)
    if alpha_mul < 1.0:
        a = np.array(sp, dtype=np.float32)
        a[:,:,3] *= alpha_mul
        sp = Image.fromarray(a.astype(np.uint8), mode='RGBA')
    px = int(center_x - new_w // 2)
    py = int(bottom_y - scale_h)
    canvas.alpha_composite(sp, dest=(px, py))
    return canvas.convert('RGB')

off_norm = Image.open(os.path.join(BG_DIR, 'office_normal.png')).convert('RGB')
off_alrm = Image.open(os.path.join(BG_DIR, 'office_alarm.png')).convert('RGB')

off_norm.save(os.path.join(CAM_DIR, 'office_normal.jpg'), quality=90)
off_alrm.save(os.path.join(CAM_DIR, 'office_alarm.jpg'), quality=90)

def paste_behind_window(bg, sprite_name, scale_h, center_x, bottom_y=525, brightness=0.62, contrast=1.40, alpha_mul=0.84, crop_bottom=0.58):
    # Window glass region in FNaF 3 Office (2000x768) is x: 815..1655, y: 155..525
    rendered = paste_char(bg, sprite_name, scale_h, center_x, bottom_y, brightness, contrast, alpha_mul, crop_bottom)
    out = bg.copy()
    win_crop = rendered.crop((815, 155, 1655, 415))
    out.paste(win_crop, (815, 155))
    return out

win_st = paste_behind_window(off_norm, 'springtrap.png', scale_h=425, center_x=1190, bottom_y=525, brightness=0.64, contrast=1.40, alpha_mul=0.86, crop_bottom=0.56)
win_st.save(os.path.join(CAM_DIR, 'office_window_springtrap.jpg'), quality=90)

door_st = paste_char(off_norm, 'springtrap.png', scale_h=680, center_x=335, bottom_y=755, brightness=0.68, contrast=1.40, crop_bottom=0.88)
door_st.save(os.path.join(CAM_DIR, 'office_door_springtrap.jpg'), quality=90)

win_pf = paste_behind_window(off_norm, 'phantom_freddy.png', scale_h=385, center_x=1140, bottom_y=525, brightness=0.68, contrast=1.42, alpha_mul=0.84, crop_bottom=0.70)
win_pf.save(os.path.join(CAM_DIR, 'office_window_phantom_freddy.jpg'), quality=90)

win_pm = paste_behind_window(off_norm, 'phantom_mangle.png', scale_h=340, center_x=1030, bottom_y=525, brightness=0.66, contrast=1.45, alpha_mul=0.86, crop_bottom=0.65)
win_pm.save(os.path.join(CAM_DIR, 'office_window_phantom_mangle.jpg'), quality=90)

left_pfx = paste_char(off_norm, 'phantom_foxy.png', scale_h=590, center_x=560, bottom_y=740, brightness=0.72, contrast=1.45, crop_bottom=0.85)
left_pfx.save(os.path.join(CAM_DIR, 'office_left_phantom_foxy.jpg'), quality=90)

inside_pp = paste_char(off_norm, 'phantom_puppet.png', scale_h=740, center_x=1120, bottom_y=768, brightness=0.82, contrast=1.50, crop_bottom=0.72)
inside_pp.save(os.path.join(CAM_DIR, 'office_inside_phantom_puppet.jpg'), quality=90)

CAM_SPECS = {
    '01': (490, 560, 410, 0.55),
    '02': (510, 440, 610, 0.60),
    '03': (470, 540, 540, 0.55),
    '04': (500, 390, 590, 0.58),
    '05': (480, 430, 580, 0.55),
    '06': (450, 415, 560, 0.52),
    '07': (490, 535, 610, 0.55),
    '08': (490, 430, 600, 0.56),
    '09': (480, 370, 585, 0.55),
    '10': (490, 520, 605, 0.56),
}

for cid, (sh, cx, by, br) in CAM_SPECS.items():
    bg = Image.open(os.path.join(BG_DIR, f'cam{cid}_empty.png')).convert('RGB')
    bg.save(os.path.join(CAM_DIR, f'cam{cid}_empty.jpg'), quality=88)

    st_cam = paste_char(bg, 'springtrap.png', scale_h=sh, center_x=cx, bottom_y=by, brightness=br, contrast=1.42, crop_bottom=0.85)
    st_cam.save(os.path.join(CAM_DIR, f'cam{cid}_springtrap.jpg'), quality=88)

    if cid in ('01', '02', '03', '05', '09', '10'):
        pbb = paste_char(bg, 'phantom_bb.png', scale_h=620, center_x=412, bottom_y=650, brightness=0.78, contrast=1.50, crop_bottom=0.65)
        pbb.save(os.path.join(CAM_DIR, f'cam{cid}_phantom_bb.jpg'), quality=88)

bg04 = Image.open(os.path.join(BG_DIR, 'cam04_empty.png')).convert('RGB')
pm04 = paste_char(bg04, 'phantom_mangle.png', scale_h=520, center_x=270, bottom_y=590, brightness=0.68, contrast=1.48, crop_bottom=0.75)
pm04.save(os.path.join(CAM_DIR, 'cam04_phantom_mangle.jpg'), quality=88)

bg07 = Image.open(os.path.join(BG_DIR, 'cam07_empty.png')).convert('RGB')
pc07 = paste_char(bg07, 'phantom_chica.png', scale_h=430, center_x=295, bottom_y=520, brightness=0.82, contrast=1.50, crop_bottom=0.68)
pc07.save(os.path.join(CAM_DIR, 'cam07_phantom_chica.jpg'), quality=88)

bg08 = Image.open(os.path.join(BG_DIR, 'cam08_empty.png')).convert('RGB')
pp08 = paste_char(bg08, 'phantom_puppet.png', scale_h=560, center_x=420, bottom_y=635, brightness=0.74, contrast=1.48, crop_bottom=0.80)
pp08.save(os.path.join(CAM_DIR, 'cam08_phantom_puppet.jpg'), quality=88)

for v_id in ('11', '12', '13', '14', '15'):
    vbg = Image.open(os.path.join(BG_DIR, f'cam{v_id}_empty.png')).convert('RGB')
    vbg.save(os.path.join(CAM_DIR, f'cam{v_id}_empty.jpg'), quality=88)
    vst = paste_char(vbg, 'springtrap.png', scale_h=540, center_x=412, bottom_y=645, brightness=0.62, contrast=1.48, crop_bottom=0.55)
    vst.save(os.path.join(CAM_DIR, f'cam{v_id}_springtrap.jpg'), quality=88)

print("Done! Total frames in assets3/cams:", len(os.listdir(CAM_DIR)))
