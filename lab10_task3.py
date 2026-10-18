import itertools
import json
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import shap
from scipy.stats import rankdata, spearmanr
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
DIR = Path(__file__).parent if "__file__" in globals() else Path(".")
(DIR / "figuras").mkdir(exist_ok=True)
(DIR / "resultados").mkdir(exist_ok=True)
R = {"seed": SEED}

# --------------------------------------------------------------------------- #
# Setup identico al Task 2.1 / 2.2
# --------------------------------------------------------------------------- #
datos = fetch_openml("credit-g", version=1, as_frame=True)
X_raw, y = datos.data.copy(), (datos.target == "good").astype(int).to_numpy()
cat_cols = [c for c in X_raw.columns if str(X_raw[c].dtype) in ("category", "object")]
num_cols = [c for c in X_raw.columns if c not in cat_cols]
enc = OrdinalEncoder(dtype=float)
X = X_raw.copy()
X[cat_cols] = enc.fit_transform(X_raw[cat_cols])
X = X.astype(float)
feats = list(X.columns)
d = len(feats)
cat_labels = dict(zip(cat_cols, enc.categories_))

idx_tr, idx_te = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=SEED)
Xtr, Xte, ytr, yte = X.values[idx_tr], X.values[idx_te], y[idx_tr], y[idx_te]
model = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=SEED).fit(Xtr, ytr)
p_te = model.predict_proba(Xte)[:, 1]

rng0 = np.random.default_rng(SEED)
bg = Xtr[rng0.choice(len(Xtr), 100, replace=False)]           # mismo fondo que el Task 2
expl = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional", model_output="raw")
phi = expl.shap_values(Xte, check_additivity=False)           # (200, 20) en log-odds

# Clientes: los dos del Task 2.2 (misma regla de seleccion) + uno con p en (0.45, 0.55)
hi, lo = np.where(p_te > 0.8)[0], np.where(p_te < 0.2)[0]
i_good = int(hi[np.argmin(np.abs(p_te[hi] - 0.9))])
i_bad = int(lo[np.argmin(np.abs(p_te[lo] - 0.1))])
mid = np.where((p_te > 0.45) & (p_te < 0.55))[0]
i_mid = int(mid[np.argmin(np.abs(p_te[mid] - 0.5))])
CLIENTS = {"good(p>0.8)": i_good, "rechazado": i_bad, "limite(p~0.5)": i_mid}
print(f"Clientes: good={i_good} (Task 2 reporta 95), rechazado={i_bad} (Task 2 reporta 134), limite={i_mid}")
R["clients"] = {k: dict(idx=v, p=float(p_te[v])) for k, v in CLIENTS.items()}
R["accuracy_check"] = dict(good_matches_task2=i_good == 95, bad_matches_task2=i_bad == 134)


def decode_val(j, v):
    return cat_labels[feats[j]][int(v)] if feats[j] in cat_labels else float(v)


# --------------------------------------------------------------------------- #
# Task 3.1  LIME tabular desde cero
# --------------------------------------------------------------------------- #
def wridge(Zb, y_, pi, lam, penalize_intercept=False):
    """Solucion cerrada w = (Z'^T Pi Z' + lam I)^-1 Z'^T Pi y. Zb lleva una columna de unos al final.
    Por defecto el INTERCEPTO NO se penaliza (se anula su entrada en la diagonal de lam*I)."""
    P = np.eye(Zb.shape[1])
    if not penalize_intercept:
        P[-1, -1] = 0.0
    A = Zb.T @ (pi[:, None] * Zb) + lam * P
    return np.linalg.solve(A, Zb.T @ (pi * y_))


def r2_w(y_, g, pi):
    """Fidelidad ponderada R^2_w = 1 - sum pi (y-g)^2 / sum pi (y-ybar_w)^2."""
    yb = (pi * y_).sum() / pi.sum()
    return 1 - (pi * (y_ - g) ** 2).sum() / (pi * (y_ - yb) ** 2).sum()


def lime(x, N=5000, K=5, lam=1.0, sigma_rel=0.75, seed=0):
    """LIME tabular. Representacion interpretable: z'_j=1 (presente, conserva x_j) o 0 (ausente,
    se reemplaza por un valor al azar de la columna j de entrenamiento)."""
    rng = np.random.default_rng(seed)
    Zp = (rng.random((N, d)) < 0.5).astype(float)                         # z' ~ Bernoulli(0.5)
    Rv = Xtr[rng.integers(0, len(Xtr), (N, d)), np.arange(d)]             # valor al azar por columna
    Z = np.where(Zp == 1, x[None, :], Rv)                                 # entrada z en el espacio original
    yv = model.predict_proba(Z)[:, 1]                                     # salida a explicar: P(good)
    D2 = (1 - Zp).sum(1)                                                  # D^2 = # variables ausentes
    sigma = sigma_rel * np.sqrt(d)
    pi = np.exp(-D2 / sigma ** 2)                                         # pi_x(z) = exp(-D^2/sigma^2)
    Zb = np.column_stack([Zp, np.ones(N)])
    w = wridge(Zb, yv, pi, lam)                                           # ajuste con las 20 variables
    r2_full = r2_w(yv, Zb @ w, pi)
    sel = np.argsort(-np.abs(w[:d]))[:K]                                  # K variables de mayor |w|
    Zs = np.column_stack([Zp[:, sel], np.ones(N)])
    ws = wridge(Zs, yv, pi, lam)                                          # re-ajuste solo con ellas
    r2 = r2_w(yv, Zs @ ws, pi)
    Nef = pi.sum() ** 2 / (pi ** 2).sum()
    return dict(w_full=w[:d], b_full=w[d], sel=sel, w_sel=ws[:-1], b_sel=ws[-1], r2=float(r2),
                r2_full=float(r2_full), Nef=float(Nef), pi=pi)


R["t31"] = {"penaliza_intercepto": False, "lam": 1.0, "N": 5000, "K": 5, "sigma_rel": 0.75}
lime_cache = {}
for name, i in CLIENTS.items():
    out = lime(Xte[i], seed=SEED)
    lime_cache[name] = out
    R["t31"][name] = dict(
        idx=i, p=float(p_te[i]), r2_K5=out["r2"], r2_full20=out["r2_full"], intercepto=float(out["b_sel"]),
        coefs=[(feats[j], decode_val(j, Xte[i][j]), float(c)) for j, c in zip(out["sel"], out["w_sel"])])

# --------------------------------------------------------------------------- #
# Task 3.2  Estabilidad (cliente rechazado)
# --------------------------------------------------------------------------- #
xb = Xte[i_bad]
Ns, SIGS, RUNS, K = [500, 2000, 5000], [0.5, 0.75, 1.5], 20, 5


def jaccard_mean(sets):
    return float(np.mean([len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2)]))


def stability(N, sig, runs=RUNS, base_seed=1000):
    sets, nefs = [], []
    for r in range(runs):
        o = lime(xb, N=N, K=K, sigma_rel=sig, seed=base_seed + r)
        sets.append(set(o["sel"].tolist()))
        nefs.append(o["Nef"])
    return jaccard_mean(sets), float(np.mean(nefs))


jac = pd.DataFrame(index=[f"N={n}" for n in Ns], columns=[f"sig={s}" for s in SIGS], dtype=float)
nef = jac.copy()
for n in Ns:
    for s in SIGS:
        jac.loc[f"N={n}", f"sig={s}"], nef.loc[f"N={n}", f"sig={s}"] = stability(n, s)
R["t32"] = dict(jaccard=jac.round(4).to_dict(), nef=nef.round(1).to_dict())
print("\nJaccard medio:\n", jac.round(3), "\n\nN_ef medio:\n", nef.round(1))

# --------------------------------------------------------------------------- #
# Task 3.3  LIME vs SHAP
# --------------------------------------------------------------------------- #
R["t33a"] = {}
for name, i in CLIENTS.items():
    w_abs = np.abs(lime_cache[name]["w_full"])
    s_abs = np.abs(phi[i])
    t3_l, t3_s = np.argsort(-w_abs)[:3], np.argsort(-s_abs)[:3]
    R["t33a"][name] = dict(spearman=float(spearmanr(w_abs, s_abs)[0]),
                           top3_lime=[feats[j] for j in t3_l], top3_shap=[feats[j] for j in t3_s],
                           overlap=len(set(t3_l) & set(t3_s)))
R["t33b"] = {}
for Kk in (3, 5, 10, 20):
    R["t33b"][f"K={Kk}"] = lime(xb, K=Kk, seed=SEED)["r2"]       # misma semilla -> mismas muestras
print("\n3.3a:", json.dumps(R["t33a"], indent=1))
print("3.3b R2_w vs K:", R["t33b"])

# --------------------------------------------------------------------------- #
# Task 3.4  Contrafactuales por muestreo (cliente rechazado)
# --------------------------------------------------------------------------- #
IMMUTABLE = ["age", "personal_status", "foreign_worker"]
ACTION = ["duration", "credit_amount", "installment_commitment", "checking_status", "savings_status"]
act_idx = [feats.index(c) for c in ACTION]
num_set = set(feats.index(c) for c in num_cols)
MAD = np.zeros(d)
for j in range(d):
    col = Xtr[:, j]
    m = np.median(np.abs(col - np.median(col)))
    MAD[j] = m if m > 0 else np.mean(np.abs(col - col.mean()))      # si MAD=0 -> desviacion absoluta media
THETA, NCAND = 0.5, 20000


def distance(x, C):
    """d(x,x') = sum_num |xj-xj'|/MAD_j + sum_cat 1[xj != xj']."""
    dist = np.zeros(len(C))
    for j in range(d):
        dist += (np.abs(C[:, j] - x[j]) / MAD[j]) if j in num_set else (C[:, j] != x[j]).astype(float)
    return dist


def counterfactuals(x, max_changes=3, seed=SEED):
    rng = np.random.default_rng(seed)
    C = np.tile(x, (NCAND, 1))
    m = rng.integers(1, max_changes + 1, NCAND)                          # entre 1 y max_changes variables
    ranks = np.argsort(np.argsort(rng.random((NCAND, len(ACTION))), 1), 1)
    chosen = ranks < m[:, None]                                          # subconjunto aleatorio por candidato
    for a, j in enumerate(act_idx):
        vals = Xtr[rng.integers(0, len(Xtr), NCAND), j]                  # valores observados en entrenamiento
        C[chosen[:, a], j] = vals[chosen[:, a]]
    changed = C != x[None, :]                                            # cambio efectivo (no solo sorteado)
    p = model.predict_proba(C)[:, 1]
    dist = distance(x, C)
    valid = (p >= THETA) & changed.any(1)
    picks, seen = [], set()
    for k in np.argsort(dist):
        if not valid[k]:
            continue
        key = frozenset(np.where(changed[k])[0].tolist())
        if key in seen:
            continue
        seen.add(key)
        picks.append(int(k))
        if len(picks) == 3:
            break
    rows = []
    for k in picks:
        ch = np.where(changed[k])[0]
        rows.append(dict(vars=[feats[j] for j in ch], original=[decode_val(j, x[j]) for j in ch],
                         nuevo=[decode_val(j, C[k, j]) for j in ch], distancia=float(dist[k]), p_good=float(p[k])))
    return dict(max_changes=max_changes, validez=float(valid.mean()), n_validos=int(valid.sum()),
                proximidad_media=float(np.mean([r["distancia"] for r in rows])) if rows else None,
                dispersion_media=float(np.mean([len(r["vars"]) for r in rows])) if rows else None,
                proximidad_media_todos_validos=float(dist[valid].mean()) if valid.any() else None,
                dispersion_media_todos_validos=float(changed[valid].sum(1).mean()) if valid.any() else None,
                cfs=rows)


cf = counterfactuals(xb, 3)
if not cf["cfs"]:                                    # regla del enunciado: si no hay validos -> maximo 5
    cf = counterfactuals(xb, 5)
    cf["nota"] = "Sin candidatos validos con max=3; se aumento a 5."
R["t34"] = cf
R["t34"]["p_original"] = float(p_te[i_bad])
print("\n3.4:", json.dumps(cf, indent=1, ensure_ascii=False))

# --------------------------------------------------------------------------- #
# Apoyo para el Task 3.5
# --------------------------------------------------------------------------- #
# (a) Jaccard con N mayor (sigma_rel = 0.75) para extrapolar a Jaccard = 0.9
extra = {}
for n in (10000, 20000, 40000):
    j_, ne_ = stability(n, 0.75, base_seed=5000)
    extra[f"N={n}"] = dict(jaccard=j_, nef=ne_)
R["t35a_extra"] = extra
print("\n3.5a corridas adicionales (sigma=0.75):", extra)

# (b) mayor discrepancia LIME vs SHAP: cliente y variable con mayor diferencia de rango
best = (None, None, -1)
for name, i in CLIENTS.items():
    rl, rs = rankdata(-np.abs(lime_cache[name]["w_full"])), rankdata(-np.abs(phi[i]))
    j = int(np.argmax(np.abs(rl - rs)))
    if abs(rl[j] - rs[j]) > best[2]:
        best = (name, j, abs(rl[j] - rs[j]))
cname, jd, _ = best
ci = CLIENTS[cname]
rs_shap = rankdata(-np.abs(phi[ci]))[jd]
sens = {}
for s in (0.25, 0.5, 0.75, 1.5):
    ws = [lime(Xte[ci], sigma_rel=s, seed=2000 + r)["w_full"] for r in range(10)]
    wm = np.mean(ws, 0)
    sens[f"sig={s}"] = dict(rank_lime=float(rankdata(-np.abs(wm))[jd]), coef=float(wm[jd]),
                            spearman=float(spearmanr(np.abs(wm), np.abs(phi[ci]))[0]))
# perfil: P(good) al variar SOLO esa variable en el cliente
grid = np.unique(np.quantile(Xtr[:, jd], np.linspace(0, 1, 40)))
Xg = np.tile(Xte[ci], (len(grid), 1))
Xg[:, jd] = grid
pg = model.predict_proba(Xg)[:, 1]
plt.figure(figsize=(5, 3.2))
plt.plot(grid, pg, "o-", ms=3)
plt.axvline(Xte[ci][jd], color="r", ls="--", label="valor del cliente")
plt.xlabel(feats[jd]); plt.ylabel("P(good)"); plt.title(f"Perfil: {cname}"); plt.legend()
plt.tight_layout(); plt.savefig(DIR / "figuras/fig_perfil_discrepancia.png", dpi=150); plt.close()
R["t35b"] = dict(cliente=cname, variable=feats[jd], rank_shap=float(rs_shap), sens_sigma=sens,
                 valor_cliente=decode_val(jd, Xte[ci][jd]),
                 perfil=dict(x=grid.tolist(), p=pg.tolist()), phi=float(phi[ci][jd]))
print("\n3.5b:", json.dumps({k: v for k, v in R["t35b"].items() if k != "perfil"}, indent=1, ensure_ascii=False))

# (d) variables con mayor |phi| del rechazado vs variables de los contrafactuales
top3 = np.argsort(-np.abs(phi[i_bad]))[:3]
info = []
for j in top3:
    g = np.unique(Xtr[:, j])
    Xg = np.tile(xb, (len(g), 1))
    Xg[:, j] = g
    pj = model.predict_proba(Xg)[:, 1]
    k = int(np.argmax(pj))
    info.append(dict(var=feats[j], phi=float(phi[i_bad][j]), inmutable=feats[j] in IMMUTABLE,
                     accionable=feats[j] in ACTION, en_contrafactuales=any(feats[j] in r["vars"] for r in cf["cfs"]),
                     MAD=float(MAD[j]), p_max_cambiando_solo_esta=float(pj.max()),
                     valor_para_p_max=decode_val(j, g[k]),
                     costo_distancia_p_max=float(distance(xb, Xg[k:k + 1])[0])))
R["t35d"] = info
print("\n3.5d:", json.dumps(info, indent=1, ensure_ascii=False))

json.dump(R, open(DIR / "resultados/task3.json", "w"), indent=2, ensure_ascii=False, default=float)
print("\nListo -> resultados/task3.json")

for n in (1500, 1800, 2000):
    j_, ne_ = stability(n, 0.75, base_seed=7000)
    print(n, round(j_, 3), round(ne_, 1))
