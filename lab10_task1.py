"""Laboratorio 10 - Task 1: Grad-CAM desde cero y CAM sobre ResNet-18 (ImageNet).

Salidas: figuras/ (PNG) y resultados/task1.json (todos los numeros del informe).
Semilla fija: 42 (el modelo es determinista; la semilla se fija por reproducibilidad).
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DIR = Path(__file__).parent
(DIR / "figuras").mkdir(exist_ok=True)
(DIR / "resultados").mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Task 1.1  Modelo y preprocesamiento
# --------------------------------------------------------------------------- #
weights = models.ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=weights).eval()
CATS = weights.meta["categories"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

geom = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
to_input = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
GRAY = 0.5  # "gris medio" en el espacio de pixeles [0, 1], antes de normalizar

# La imagen (c) se recorta antes del preprocesamiento estandar para que el sofa
# quede dentro del recorte central de 224x224 (el original es muy apaisado).
sofa = Image.open(DIR / "imagenes/img_c_sofa_original.jpg").convert("RGB")
sofa.crop((160, 0, 640, 427)).save(DIR / "imagenes/img_c_sofa.jpg", quality=95)
# Igual para (b): recorte cuadrado previo para que el gato (borde izquierdo) y el perro
# sobrevivan al recorte central de 224x224.
Image.open(DIR / "imagenes/img_b_gato_perro_original.jpg").convert("RGB").crop((85, 0, 725, 640)).save(
    DIR / "imagenes/img_b_gato_perro.jpg", quality=95)

IMAGES = {
    "a": ("imagenes/img_a_golden.jpg", "(a) Objeto centrado"),
    "b": ("imagenes/img_b_gato_perro.jpg", "(b) Dos objetos"),
    "c": ("imagenes/img_c_sofa.jpg", "(c) Fuera de contexto"),
}
crop = {}   # imagen 224x224 en [0,1], tensor (3,224,224)
for k, (path, _) in IMAGES.items():
    img224 = geom(Image.open(DIR / path).convert("RGB"))
    crop[k] = transforms.functional.to_tensor(img224)


def norm_input(img01):
    """(3,224,224) en [0,1] -> tensor normalizado (1,3,224,224)."""
    return transforms.functional.normalize(img01, MEAN, STD)[None]


# Recuadro del sofa en coordenadas de la imagen (c) pre-recortada (480x427)
# x en [330-160, 545-160], y en [253, 372]; se transforma al recorte 224x224.
sc = 256 / 427                       # factor de Resize(256) sobre el lado corto (427)
left, top = (480 * sc - 224) / 2, (427 * sc - 224) / 2
bx0, bx1 = (330 - 160) * sc - left, (545 - 160) * sc - left
by0, by1 = 253 * sc - top, 372 * sc - top
BBOX = [int(round(bx0)), int(round(by0)), int(round(bx1)), int(round(by1))]  # x0,y0,x1,y1

# --------------------------------------------------------------------------- #
# Task 1.2  Grad-CAM desde cero (hooks sobre layer4)
# --------------------------------------------------------------------------- #
store = {}
model.layer4.register_forward_hook(lambda m, i, o: store.__setitem__("A", o))               # A^k_ij
model.layer4.register_full_backward_hook(lambda m, gi, go: store.__setitem__("G", go[0]))   # dy^c/dA


def gradcam(x, c):
    """Devuelve alpha_k^c, L^c (7x7, con y sin ReLU), mapa 224x224 normalizado, A y logits."""
    model.zero_grad()
    logits = model(x)
    logits[0, c].backward()                       # se retropropaga el LOGIT y^c, no el softmax
    A = store["A"][0].detach()                    # (512,7,7)
    G = store["G"][0].detach()                    # (512,7,7)
    alpha = G.mean(dim=(1, 2))                    # alpha_k^c = promedio espacial del gradiente
    raw = (alpha[:, None, None] * A).sum(0)       # sum_k alpha_k A^k   (sin ReLU)
    L = F.relu(raw)                               # L^c = ReLU(sum_k alpha_k A^k)
    up = F.interpolate(L[None, None], size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    up = (up - up.min()) / (up.max() - up.min() + 1e-12)   # normalizacion min-max
    return dict(alpha=alpha, raw=raw, L=L, up=up, A=A, logits=logits.detach()[0])


def minmax(m):
    return (m - m.min()) / (m.max() - m.min() + 1e-12)


# --------------------------------------------------------------------------- #
# Ejecucion
# --------------------------------------------------------------------------- #
R = {"seed": SEED, "bbox_c_224": BBOX}
CAT_IDX = set(range(281, 286))          # gatos domesticos de ImageNet
DOG_IDX = set(range(151, 269))          # perros de ImageNet
W, bias = model.fc.weight.detach(), model.fc.bias.detach()
Z = 49
res = {}

for k in IMAGES:
    x = norm_input(crop[k])
    with torch.no_grad():
        probs = F.softmax(model(x), 1)[0]
    top5 = probs.topk(5)
    c = int(top5.indices[0])
    cls = {"top1": c}
    if k == "b":   # clase del segundo objeto: la mejor del grupo (gato/perro) que no contiene a top-1
        other = DOG_IDX if c in CAT_IDX else CAT_IDX
        cls["second"] = max(other, key=lambda i: probs[i].item())
    r = {"top5": [(CATS[int(i)], int(i), float(p)) for p, i in zip(top5.values, top5.indices)],
         "classes": {n: (CATS[i], i, float(probs[i])) for n, i in cls.items()}}

    gc = {n: gradcam(x, i) for n, i in cls.items()}
    res[k] = dict(x=x, probs=probs, cls=cls, gc=gc)

    # ---------------- Task 1.3  CAM, descomposicion y relacion con Grad-CAM ------------- #
    g = gc["top1"]
    A = g["A"]
    M = (W[c][:, None, None] * A).sum(0)                       # M_ij^c = sum_k w_k^c A_ij^k
    logit = float(g["logits"][c])
    recon = float(M.mean() + bias[c])                          # (1/Z) sum_ij M_ij + b_c
    err_decomp = abs(recon - logit)
    L_nr = g["raw"]                                            # L^c sin ReLU ni normalizacion
    err_gradcam_M = float((Z * L_nr - M).abs().max())          # |Z L^c - M^c| maximo sobre 49 posiciones
    err_norm = float((minmax(F.relu(L_nr)) - minmax(F.relu(M))).abs().max())
    r["task13"] = dict(logit=logit, recon=recon, err_decomp=err_decomp,
                       err_ZL_vs_M=err_gradcam_M, err_relu_minmax=err_norm,
                       M_range=[float(M.min()), float(M.max())])
    res[k]["M"] = M

    # ---------------- Task 1.4a (solo imagen a) ----------------------------------------- #
    if k == "a":
        wk = W[c]
        idx = wk.topk(5).indices
        rows = []
        for j in idx.tolist():
            a_j, t_j = float(g["alpha"][j]), float(wk[j] / Z)
            rows.append(dict(canal=j, w=float(wk[j]), alpha=a_j, w_sobre_Z=t_j,
                             err_rel=abs(a_j - t_j) / abs(t_j)))
        r["task14a"] = rows
        r["task14a_all"] = dict(max_abs_err=float((g["alpha"] - wk / Z).abs().max()),
                                max_rel_err=float(((g["alpha"] - wk / Z).abs() / (wk / Z).abs()).max()),
                                dtype=str(g["alpha"].dtype))

    # ---------------- Task 1.4b (solo imagen b) ----------------------------------------- #
    if k == "b":
        u1, u2 = gc["top1"]["up"].flatten(), gc["second"]["up"].flatten()
        n20 = int(np.ceil(0.2 * u1.numel()))
        s1 = set(torch.argsort(u1, descending=True, stable=True)[:n20].tolist())
        s2 = set(torch.argsort(u2, descending=True, stable=True)[:n20].tolist())
        # Variante a nivel de celda 7x7 (top 20% = 10 de 49 celdas) como complemento
        c1, c2 = gc["top1"]["L"].flatten(), gc["second"]["L"].flatten()
        t1 = set(torch.argsort(c1, descending=True, stable=True)[:10].tolist())
        t2 = set(torch.argsort(c2, descending=True, stable=True)[:10].tolist())
        r["task14b"] = dict(iou_224=len(s1 & s2) / len(s1 | s2), n_pix=n20,
                            iou_7x7=len(t1 & t2) / len(t1 | t2),
                            spearman_7x7=None)
        from scipy.stats import spearmanr
        r["task14b"]["spearman_7x7"] = float(spearmanr(c1.numpy(), c2.numpy())[0])

    # ---------------- Task 1.1 (c) y Task 1.4c ------------------------------------------ #
    if k == "c":
        x0, y0, x1, y1 = BBOX
        up = gc["top1"]["up"]
        inside = float(up[y0:y1, x0:x1].sum() / up.sum())
        area = (x1 - x0) * (y1 - y0) / 224 ** 2
        img_out = torch.full_like(crop[k], GRAY)              # fuera del recuadro -> gris
        img_out[:, y0:y1, x0:x1] = crop[k][:, y0:y1, x0:x1]
        img_in = crop[k].clone()                              # prueba complementaria: objeto -> gris
        img_in[:, y0:y1, x0:x1] = GRAY
        with torch.no_grad():
            p_full = float(probs[c])
            p_out = float(F.softmax(model(norm_input(img_out)), 1)[0, c])
            p_in = float(F.softmax(model(norm_input(img_in)), 1)[0, c])
            top_out = int(F.softmax(model(norm_input(img_out)), 1)[0].argmax())
            top_in = int(F.softmax(model(norm_input(img_in)), 1)[0].argmax())
        r["task14c"] = dict(bbox=BBOX, mass_inside=inside, area_fraction=area,
                            ratio=inside / area, p_full=p_full, p_bg_gray=p_out,
                            p_obj_gray=p_in, top_bg_gray=CATS[top_out], top_obj_gray=CATS[top_in])
        res[k]["img_out"], res[k]["img_in"] = img_out, img_in
    R[k] = r

# Task 1.4a (complemento): el mismo calculo en doble precision para ver el piso numerico
model.double()
xa = res["a"]["x"].double()
g64 = gradcam(xa, res["a"]["cls"]["top1"])
ca = res["a"]["cls"]["top1"]
w64 = model.fc.weight.detach()[ca]
rel64 = ((g64["alpha"] - w64 / Z).abs() / (w64 / Z).abs())
R["a"]["task14a_float64"] = dict(max_rel_err=float(rel64.max()), max_abs_err=float((g64["alpha"] - w64 / Z).abs().max()),
                                 top5_rel_err=[float(rel64[j]) for j in w64.topk(5).indices.tolist()])
model.float()

json.dump(R, open(DIR / "resultados/task1.json", "w"), indent=2, ensure_ascii=False)

# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def overlay(ax, img01, heat, title):
    ax.imshow(img01.permute(1, 2, 0).numpy())
    ax.imshow(heat.numpy(), cmap="jet", alpha=0.5, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


panels = [("a", "top1"), ("b", "top1"), ("b", "second"), ("c", "top1")]
fig, axs = plt.subplots(2, 4, figsize=(12, 6.6))
for j, (k, n) in enumerate(panels):
    cidx = res[k]["cls"][n]
    axs[0, j].imshow(crop[k].permute(1, 2, 0).numpy())
    axs[0, j].set_title(f"({k}) entrada 224x224", fontsize=8)
    axs[0, j].axis("off")
    if k == "c":
        x0, y0, x1, y1 = BBOX
        axs[0, j].add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="lime", lw=2))
    overlay(axs[1, j], crop[k], res[k]["gc"][n]["up"],
            f"Grad-CAM: {CATS[cidx][:24]}\np={float(res[k]['probs'][cidx]):.3f}")
plt.tight_layout(h_pad=2.5)
plt.savefig(DIR / "figuras/fig_gradcam.png", dpi=150)
plt.close()

# Figura de la prueba causal (imagen c)
fig, axs = plt.subplots(1, 3, figsize=(8.5, 3))
for ax, im, t in zip(axs, [crop["c"], res["c"]["img_out"], res["c"]["img_in"]],
                     [f"original  p={R['c']['task14c']['p_full']:.3f}",
                      f"fondo gris  p={R['c']['task14c']['p_bg_gray']:.3f}",
                      f"objeto gris  p={R['c']['task14c']['p_obj_gray']:.3f}"]):
    ax.imshow(im.permute(1, 2, 0).numpy())
    ax.set_title(t, fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig(DIR / "figuras/fig_causal.png", dpi=150)
plt.close()

# --------------------------------------------------------------------------- #
# Resumen en consola
# --------------------------------------------------------------------------- #
print(json.dumps(R, indent=1, ensure_ascii=False))
