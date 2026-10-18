"""Laboratorio 10 - Task 4: validacion de Grad-CAM (deletion / insertion y aleatorizacion en cascada).

Salidas: figuras/ (PNG) y resultados/task4.json. Semilla fija: 42.
Grad-CAM es el mismo del Task 1.2 (hooks sobre layer4, escrito a mano; sin pytorch-grad-cam ni captum).
"""
import copy
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from scipy.stats import spearmanr
from torchvision import models, transforms
from torchvision.transforms import functional as TF

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DIR = Path(__file__).parent
(DIR / "figuras").mkdir(exist_ok=True)
(DIR / "resultados").mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Setup identico al Task 1.1 (las imagenes (b) y (c) ya estan pre-recortadas por el Task 1)
# --------------------------------------------------------------------------- #
weights = models.ResNet18_Weights.IMAGENET1K_V1
cnn = models.resnet18(weights=weights).eval()
CATS = weights.meta["categories"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
geom = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
GRAY = 0.5                                   # gris medio en el espacio de pixeles [0, 1]
IMGS = {"a": "imagenes/img_a_golden.jpg", "b": "imagenes/img_b_gato_perro.jpg", "c": "imagenes/img_c_sofa.jpg"}
img01 = {k: TF.to_tensor(geom(PILImage.open(DIR / p).convert("RGB"))) for k, p in IMGS.items()}   # (3,224,224)
HW = 224 * 224
R4 = {"seed": SEED}


def to_x(im01):
    """Imagen(es) en [0,1] -> entrada normalizada de ImageNet."""
    return TF.normalize(im01, MEAN, STD)


def gradcam_net(net, im01, c):
    """Grad-CAM del Task 1.2 sobre cualquier red con layer4 (hooks temporales).
    alpha_k^c = promedio espacial de dy^c/dA^k ; L^c = ReLU(sum_k alpha_k^c A^k) ; ampliado y min-max."""
    st = {}
    h1 = net.layer4.register_forward_hook(lambda m, i, o: st.__setitem__("A", o))               # A^k_ij
    h2 = net.layer4.register_full_backward_hook(lambda m, gi, go: st.__setitem__("G", go[0]))   # dy^c/dA
    net.zero_grad()
    logits = net(to_x(im01)[None])
    logits[0, c].backward()                       # se retropropaga el LOGIT y^c (antes del softmax)
    h1.remove(); h2.remove()
    A, G = st["A"][0].detach(), st["G"][0].detach()
    alpha = G.mean(dim=(1, 2))                    # alpha_k^c
    raw = (alpha[:, None, None] * A).sum(0)       # sum_k alpha_k^c A^k (sin ReLU)
    L = F.relu(raw)                               # L^c (7x7)
    up = F.interpolate(L[None, None], size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    up = (up - up.min()) / (up.max() - up.min() + 1e-12)
    return dict(alpha=alpha, raw=raw, L=L, up=up, logits=logits.detach()[0])


@torch.no_grad()
def prob_class(net, ims01, c, bs=32):
    """P(clase c) para un lote de imagenes en [0,1]."""
    return torch.cat([F.softmax(net(to_x(ims01[i:i + bs])), 1)[:, c] for i in range(0, len(ims01), bs)]).numpy()


top1 = {}
maps = {}
for k in IMGS:
    with torch.no_grad():
        top1[k] = int(cnn(to_x(img01[k])[None]).argmax())
    maps[k] = gradcam_net(cnn, img01[k], top1[k])
R4["clases"] = {k: dict(clase=CATS[c], idx=c) for k, c in top1.items()}

# --------------------------------------------------------------------------- #
# Task 4.1  Deletion e insertion (P = 20 pasos, 10 ordenes aleatorios)
# --------------------------------------------------------------------------- #
P, N_RAND = 20, 10
BLUR_SIGMA, BLUR_KS = 10.0, 61               # desenfoque gaussiano sigma = 10 px (nucleo 2*ceil(3 sigma)+1)
rng4 = np.random.default_rng(SEED)


def map_order(up):
    """Pixeles de mayor a menor valor del mapa. Los empates (zonas donde el mapa vale 0 tras la ReLU)
    se ordenan al azar, para no imponer un orden de barrido por filas."""
    perm = torch.from_numpy(rng4.permutation(HW))
    o = torch.argsort(up.flatten()[perm], descending=True, stable=True)
    return perm[o]


def perturb_curve(base01, src01, order, c):
    """c_k para k = 0..P: la fraccion k/P de pixeles con mayor prioridad (segun `order`) se toma de src01
    y el resto de base01. Deletion: base = imagen, src = gris. Insertion: base = desenfocada, src = imagen."""
    ims = []
    for k in range(P + 1):
        m = torch.zeros(HW, dtype=torch.bool)
        m[order[: int(round(k / P * HW))]] = True
        ims.append(torch.where(m.view(1, 224, 224), src01, base01))
    return prob_class(cnn, torch.stack(ims), c)


def auc(curve):
    """AUC = (1/P) sum_{k=1}^{P} c_k (el punto k=0 no entra)."""
    return float(np.mean(curve[1:]))


R4["t41"] = {}
curves = {}
for k in IMGS:
    x, c = img01[k], top1[k]
    gray = torch.full_like(x, GRAY)
    blur = TF.gaussian_blur(x, kernel_size=BLUR_KS, sigma=BLUR_SIGMA)
    o_map = map_order(maps[k]["up"])
    d_map, i_map = perturb_curve(x, gray, o_map, c), perturb_curve(blur, x, o_map, c)
    d_rnd, i_rnd = [], []
    for _ in range(N_RAND):
        o_r = torch.from_numpy(rng4.permutation(HW))
        d_rnd.append(perturb_curve(x, gray, o_r, c))
        i_rnd.append(perturb_curve(blur, x, o_r, c))
    d_rnd, i_rnd = np.array(d_rnd), np.array(i_rnd)
    ad_r, ai_r = np.array([auc(v) for v in d_rnd]), np.array([auc(v) for v in i_rnd])
    ad, ai = auc(d_map), auc(i_map)
    R4["t41"][k] = dict(
        clase=CATS[c], p_original=float(d_map[0]), p_desenfocada=float(i_map[0]),
        frac_mapa_cero=float((maps[k]["up"] == 0).float().mean()),
        AUC_del=ad, AUC_del_rand_mean=float(ad_r.mean()), AUC_del_rand_std=float(ad_r.std(ddof=1)),
        AUC_ins=ai, AUC_ins_rand_mean=float(ai_r.mean()), AUC_ins_rand_std=float(ai_r.std(ddof=1)),
        dif_del=float(ad_r.mean() - ad), dif_ins=float(ai - ai_r.mean()),          # > 0: mejor que el azar
        z_del=float((ad_r.mean() - ad) / ad_r.std(ddof=1)), z_ins=float((ai - ai_r.mean()) / ai_r.std(ddof=1)),
        curva_del=d_map.tolist(), curva_ins=i_map.tolist(),
        curva_del_rand_mean=d_rnd.mean(0).tolist(), curva_ins_rand_mean=i_rnd.mean(0).tolist())
    curves[k] = dict(d_map=d_map, i_map=i_map, d_rnd=d_rnd, i_rnd=i_rnd)

    # Task 4.3a (solo imagen a): deletion con referencia desenfocada en lugar de gris
    if k == "a":
        db_map = perturb_curve(x, blur, o_map, c)
        db_rnd = np.array([perturb_curve(x, blur, torch.from_numpy(rng4.permutation(HW)), c) for _ in range(N_RAND)])
        adb_r = np.array([auc(v) for v in db_rnd])
        R4["t43a_blur"] = dict(AUC_del_gris=ad, AUC_del_blur=auc(db_map),
                               AUC_del_blur_rand_mean=float(adb_r.mean()), AUC_del_blur_rand_std=float(adb_r.std(ddof=1)),
                               z_del_blur=float((adb_r.mean() - auc(db_map)) / adb_r.std(ddof=1)),
                               curva_del_blur=db_map.tolist(), curva_del_blur_rand_mean=db_rnd.mean(0).tolist())
        curves[k]["db_map"], curves[k]["db_rnd"] = db_map, db_rnd

tabla41 = pd.DataFrame({k: {m: v for m, v in r.items() if not m.startswith("curva") and m != "clase"}
                        for k, r in R4["t41"].items()}).T
print("Task 4.1\n", tabla41.round(4).to_string())
print("Task 4.3a (imagen a, deletion gris vs desenfocada):",
      {m: round(v, 4) for m, v in R4["t43a_blur"].items() if not m.startswith("curva")})

frac = np.arange(P + 1) / P
fig, axs = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True, sharey=True)
for j, k in enumerate(IMGS):
    for i, (key_m, key_r, nombre) in enumerate([("d_map", "d_rnd", "Deletion (gris)"), ("i_map", "i_rnd", "Insertion (desde desenfocada)")]):
        ax, cm, cr = axs[i, j], curves[k][key_m], curves[k][key_r]
        ax.plot(frac, cm, "o-", ms=3, label="orden Grad-CAM")
        ax.plot(frac, cr.mean(0), "s--", ms=3, color="gray", label="orden aleatorio (media)")
        ax.fill_between(frac, cr.mean(0) - cr.std(0, ddof=1), cr.mean(0) + cr.std(0, ddof=1), color="gray", alpha=0.25)
        if k == "a" and i == 0:
            ax.plot(frac, curves[k]["db_map"], "^-", ms=3, color="tab:green", label="Grad-CAM, ref. desenfocada")
        auc_m = R4["t41"][k]["AUC_del" if i == 0 else "AUC_ins"]
        ax.set_title(f"({k}) {CATS[top1[k]][:20]} - {nombre}\nAUC mapa = {auc_m:.3f}", fontsize=8)
        ax.set_xlabel("fraccion de pixeles k/P", fontsize=8)
        ax.set_ylabel("P(clase)", fontsize=8)
        ax.legend(fontsize=6)
plt.tight_layout()
plt.savefig(DIR / "figuras/fig_deletion_insertion.png", dpi=150)
plt.close()

# --------------------------------------------------------------------------- #
# Task 4.2  Aleatorizacion de parametros en cascada
# --------------------------------------------------------------------------- #
# Etapas acumulativas: fc; + layer4; + layer3; + layer2; + layer1; + conv1. Cada modulo dentro del bloque
# (Conv2d, BatchNorm2d, Linear) vuelve a la inicializacion por defecto de PyTorch con reset_parameters().
# El mapa se calcula SIEMPRE para la clase top-1 del modelo original, no para la nueva prediccion.
STAGES = ["fc", "layer4", "layer3", "layer2", "layer1", "conv1"]


def reinit(block):
    for m in block.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()


def spearman_or_none(u, v):
    """Spearman entre dos mapas 7x7; None si alguno es constante (p. ej. todo cero tras la ReLU)."""
    u, v = u.flatten().numpy(), v.flatten().numpy()
    if np.ptp(u) == 0 or np.ptp(v) == 0:
        return None
    return float(spearmanr(u, v)[0])


torch.manual_seed(SEED)
net = copy.deepcopy(cnn).eval()
casc = {k: [] for k in IMGS}
casc_maps = {k: [maps[k]["L"]] for k in IMGS}
for s in STAGES:
    reinit(getattr(net, s))
    for k in IMGS:
        c = top1[k]
        g = gradcam_net(net, img01[k], c)
        p_new = F.softmax(g["logits"], 0)
        row = dict(etapa=s, spearman=spearman_or_none(maps[k]["L"], g["L"]),
                   spearman_sin_relu=spearman_or_none(maps[k]["raw"], g["raw"]),
                   celdas_positivas=int((g["L"] > 0).sum()),
                   p_clase_original=float(p_new[c]), nueva_top1=CATS[int(p_new.argmax())])
        if s == "fc":   # en la primera etapa alpha^c debe seguir siendo w^c/Z, ahora con los pesos aleatorios
            w_new = net.fc.weight.detach()[c]
            row["max_err_rel_alpha_vs_w_Z"] = float(((g["alpha"] - w_new / 49).abs() / (w_new / 49).abs()).max())
        casc[k].append(row)
        casc_maps[k].append(g["L"])
R4["t42"] = casc
tabla42 = pd.DataFrame({k: {r["etapa"]: r["spearman"] for r in v} for k, v in casc.items()}).T[STAGES]
print("\nTask 4.2 Spearman (mapa 7x7 con ReLU) vs mapa original:\n", tabla42.round(3).to_string())
print("Spearman sin ReLU:\n", pd.DataFrame({k: {r["etapa"]: r["spearman_sin_relu"] for r in v}
                                             for k, v in casc.items()}).T[STAGES].round(3).to_string())

# Task 4.3b: primera etapa con Spearman < 0.3 (un mapa nulo cuenta como sin correlacion)
R4["t43b"] = {k: next((r["etapa"] for r in v if r["spearman"] is None or r["spearman"] < 0.3), None)
              for k, v in casc.items()}
print("Primera etapa con Spearman < 0.3:", R4["t43b"])

fig, axs = plt.subplots(3, 7, figsize=(13, 7.4))
for i, k in enumerate(IMGS):
    for j, Lm in enumerate(casc_maps[k]):
        ax = axs[i, j]
        rng_m = float(Lm.max() - Lm.min())
        ax.imshow(((Lm - Lm.min()) / rng_m).numpy() if rng_m > 0 else Lm.numpy(), cmap="jet", vmin=0, vmax=1,
                  interpolation="nearest")
        if j == 0:
            ax.set_title(f"({k}) original", fontsize=8)
        else:
            rho = casc[k][j - 1]["spearman"]
            ax.set_title(f"+{STAGES[j - 1]}\nrho={'nulo' if rho is None else f'{rho:.2f}'}", fontsize=8)
        ax.axis("off")
plt.tight_layout()
plt.savefig(DIR / "figuras/fig_cascada.png", dpi=150)
plt.close()

json.dump(R4, open(DIR / "resultados/task4.json", "w"), indent=2, ensure_ascii=False)
print("\nListo -> resultados/task4.json")
