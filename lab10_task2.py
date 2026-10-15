"""Laboratorio 10 - Task 2: SHAP sobre German Credit (gradient boosting).

Salidas: figuras/ (PNG) y resultados/task2.json (todos los numeros del informe).
Semilla aleatoria fija: 42.
"""
import json
import math
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import shap
from scipy.stats import rankdata, spearmanr
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
rng = np.random.default_rng(SEED)
DIR = Path(__file__).parent
(DIR / "figuras").mkdir(exist_ok=True)
(DIR / "resultados").mkdir(exist_ok=True)
R = {"seed": SEED}
sigmoid = lambda z: 1 / (1 + np.exp(-z))

# --------------------------------------------------------------------------- #
# Task 2.1  Datos y modelo
# --------------------------------------------------------------------------- #
datos = fetch_openml("credit-g", version=1, as_frame=True)
X_raw, y = datos.data.copy(), (datos.target == "good").astype(int).to_numpy()   # good=1, bad=0
cat_cols = [c for c in X_raw.columns if str(X_raw[c].dtype) in ("category", "object")]
num_cols = [c for c in X_raw.columns if c not in cat_cols]
enc = OrdinalEncoder(dtype=float)       # codigos arbitrarios (se tratan como numericos)
X = X_raw.copy()
X[cat_cols] = enc.fit_transform(X_raw[cat_cols])
X = X.astype(float)
feats = list(X.columns)
d = len(feats)
cat_labels = dict(zip(cat_cols, enc.categories_))
R["n"], R["n_num"], R["n_cat"] = len(X), len(num_cols), len(cat_cols)
R["prop_good"] = float(y.mean())

idx_tr, idx_te = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=SEED)
Xtr, Xte, ytr, yte = X.values[idx_tr], X.values[idx_te], y[idx_tr], y[idx_te]

N_TREES, DEPTH = 200, 3


def fit(Xa, ya):
    return GradientBoostingClassifier(n_estimators=N_TREES, max_depth=DEPTH, random_state=SEED).fit(Xa, ya)


model = fit(Xtr, ytr)
f_raw = model.decision_function              # salida en log-odds
p_te = model.predict_proba(Xte)[:, 1]
R["t21"] = dict(acc=float(accuracy_score(yte, model.predict(Xte))), auc=float(roc_auc_score(yte, p_te)),
                n_train=len(idx_tr), n_test=len(idx_te))

bg_idx = rng.choice(len(Xtr), 100, replace=False)       # fondo de 100 instancias de entrenamiento
bg = Xtr[bg_idx]

# --------------------------------------------------------------------------- #
# Task 2.2  SHAP intervencional sobre log-odds
# --------------------------------------------------------------------------- #
expl = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional", model_output="raw")
phi = expl.shap_values(Xte, check_additivity=False)
base = float(np.ravel(expl.expected_value)[0])
err_eff = np.abs(base + phi.sum(1) - f_raw(Xte))
R["t22a"] = dict(max_err=float(err_eff.max()), mean_err=float(err_eff.mean()))
R["t22b"] = dict(base_value=base, mean_logodds_bg=float(f_raw(bg).mean()), diff=abs(base - float(f_raw(bg).mean())))

I = np.abs(phi).mean(0)                                           # I_j = 1/N sum_n |phi_j^(n)|
perm = permutation_importance(model, Xte, yte, scoring="roc_auc", n_repeats=10, random_state=SEED)
Ip = perm.importances_mean
top = lambda v: [(feats[j], float(v[j])) for j in np.argsort(-v)[:5]]
R["t22c"] = dict(top_shap=top(I), top_perm=top(Ip), spearman=float(spearmanr(I, Ip)[0]),
                 I={feats[j]: float(I[j]) for j in range(d)}, perm={feats[j]: float(Ip[j]) for j in range(d)})


def decode(row):
    """Valores de una fila con las categoricas decodificadas (solo para mostrar)."""
    out = []
    for j, c in enumerate(feats):
        out.append(cat_labels[c][int(row[j])] if c in cat_labels else float(row[j]))
    return out


# clientes: p(good) mas cercana a 0.9 entre los > 0.8, y a 0.1 entre los < 0.2
cand_hi, cand_lo = np.where(p_te > 0.8)[0], np.where(p_te < 0.2)[0]
i_good = int(cand_hi[np.argmin(np.abs(p_te[cand_hi] - 0.9))])
i_bad = int(cand_lo[np.argmin(np.abs(p_te[cand_lo] - 0.1))])
R["clients"] = {"good": i_good, "bad": i_bad, "n_hi": int(len(cand_hi)), "n_lo": int(len(cand_lo))}
for name, i in [("good", i_good), ("bad", i_bad)]:
    o = np.argsort(-np.abs(phi[i]))[:3]
    R["clients"][name + "_info"] = dict(
        p=float(p_te[i]), f=float(f_raw(Xte[i:i + 1])[0]), y_true=int(yte[i]),
        top3=[(feats[j], decode(Xte[i])[j], float(phi[i, j])) for j in o])
    ex = shap.Explanation(values=phi[i], base_values=base, data=np.array(decode(Xte[i]), dtype=object),
                          feature_names=feats)
    plt.figure()
    shap.plots.waterfall(ex, max_display=10, show=False)
    plt.gcf().set_size_inches(7, 4.2)
    plt.title(f"p(good)={p_te[i]:.3f}", fontsize=10)
    plt.savefig(DIR / f"figuras/fig_waterfall_{name}.png", dpi=150, bbox_inches="tight")
    plt.close("all")

plt.figure()
shap.plots.beeswarm(shap.Explanation(values=phi, base_values=np.full(len(phi), base), data=Xte, feature_names=feats),
                    max_display=12, show=False)
plt.gcf().set_size_inches(7, 4.8)
plt.savefig(DIR / "figuras/fig_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close("all")

# --------------------------------------------------------------------------- #
# Task 2.3  Variable redundante
# --------------------------------------------------------------------------- #
num_idx = [feats.index(c) for c in num_cols]
j0 = max(num_idx, key=lambda j: I[j])
sx = Xtr[:, j0].std()
rng_noise = np.random.default_rng(SEED)
x_dup = X.values[:, j0] + 0.05 * sx * rng_noise.standard_normal(len(X))       # x + 0.05 sigma_x eps
X21 = np.column_stack([X.values, x_dup])
feats21 = feats + [feats[j0] + "_copia"]
Xtr21, Xte21 = X21[idx_tr], X21[idx_te]
model21 = fit(Xtr21, ytr)
bg21 = Xtr21[bg_idx]
cfg = {}
e_int = shap.TreeExplainer(model21, data=bg21, feature_perturbation="interventional", model_output="raw")
e_tp = shap.TreeExplainer(model21, feature_perturbation="tree_path_dependent", model_output="raw")
for name, e in [("intervencional", e_int), ("tree_path_dependent", e_tp)]:
    ph = e.shap_values(Xte21, check_additivity=False)
    cfg[name] = dict(orig=float(np.abs(ph[:, j0]).mean()), copia=float(np.abs(ph[:, d]).mean()),
                     suma=float(np.abs(ph[:, j0] + ph[:, d]).mean()),
                     corr_phi=float(np.corrcoef(ph[:, j0], ph[:, d])[0, 1]))
R["t23"] = dict(var=feats[j0], I_sin_copia=float(I[j0]), corr_x_copia=float(np.corrcoef(Xte[:, j0], Xte21[:, d])[0, 1]),
                auc21=float(roc_auc_score(yte, model21.predict_proba(Xte21)[:, 1])), configs=cfg)

# --------------------------------------------------------------------------- #
# Task 2.4  KernelSHAP vs TreeSHAP
# --------------------------------------------------------------------------- #
sel = rng.choice(len(Xte), 10, replace=False)
Xk = Xte[sel]
rows = []
for nb in (10, 50, 100):
    bgk = bg[:nb]
    t0 = time.perf_counter()
    ks = shap.KernelExplainer(f_raw, bgk, link="identity")
    pk = ks.shap_values(Xk, nsamples=2000, l1_reg=0, silent=True)
    t_kernel = time.perf_counter() - t0
    t0 = time.perf_counter()
    tr = shap.TreeExplainer(model, data=bgk, feature_perturbation="interventional", model_output="raw")
    pt = tr.shap_values(Xk, check_additivity=False)
    t_tree = time.perf_counter() - t0
    rows.append(dict(bg=nb, mae=float(np.abs(pk - pt).mean()), max_err=float(np.abs(pk - pt).max()),
                     t_kernel=t_kernel, t_kernel_inst=t_kernel / len(Xk), t_tree=t_tree,
                     eff_err=float(np.abs(ks.expected_value + pk.sum(1) - f_raw(Xk)).max())))
R["t24"] = rows

# --------------------------------------------------------------------------- #
# Task 2.5  Shapley exacto escrito a mano (seis variables numericas)
# --------------------------------------------------------------------------- #
six = ["duration", "credit_amount", "installment_commitment", "residence_since", "age", "existing_credits"]
six_idx = [feats.index(c) for c in six]
n6 = len(six)
Xtr6, Xte6 = Xtr[:, six_idx], Xte[:, six_idx]
model6 = fit(Xtr6, ytr)
bg6 = Xtr6[np.random.default_rng(SEED).choice(len(Xtr6), 50, replace=False)]
f6 = model6.decision_function
R["t25_model"] = dict(acc=float(accuracy_score(yte, model6.predict(Xte6))),
                      auc=float(roc_auc_score(yte, model6.predict_proba(Xte6)[:, 1])))


def v_all(x, bg_, f):
    """v(S) para las 2^n coaliciones: promedio de f sobre el fondo con x_S fijo y el resto del fondo."""
    n, nb = len(x), len(bg_)
    masks = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(bool)      # (2^n, n): mascara de S
    Z = np.where(masks[:, None, :], x[None, None, :], bg_[None, :, :])            # (2^n, nb, n)
    return f(Z.reshape(-1, n)).reshape(1 << n, nb).mean(1)


def shapley_exact(v, n):
    """phi_i = sum_{S subset N\\{i}} |S|!(n-|S|-1)!/n! * [v(S u {i}) - v(S)]."""
    phi_ = np.zeros(n)
    for i in range(n):
        for S in range(1 << n):
            if S & (1 << i):
                continue
            s = bin(S).count("1")
            w = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
            phi_[i] += w * (v[S | (1 << i)] - v[S])
    return phi_


sel5 = rng.choice(len(Xte6), 5, replace=False)
e6 = shap.TreeExplainer(model6, data=bg6, feature_perturbation="interventional", model_output="raw")
mine, ref, eff = [], [], []
for i in sel5:
    v = v_all(Xte6[i], bg6, f6)
    ph = shapley_exact(v, n6)
    mine.append(ph)
    ref.append(e6.shap_values(Xte6[i:i + 1], check_additivity=False)[0])
    eff.append(abs(ph.sum() - (v[-1] - v[0])))          # v[-1]=v(N)=f(x), v[0]=v(empty)=media de f en el fondo
mine, ref = np.array(mine), np.array(ref)
R["t25"] = dict(idx=[int(i) for i in sel5], max_err_vs_tree=float(np.abs(mine - ref).max()),
                max_eff_err=float(max(eff)), phi_mine_0=mine[0].tolist(), phi_tree_0=ref[0].tolist(),
                v_N_minus_v_empty_0=float(v_all(Xte6[sel5[0]], bg6, f6)[-1] - v_all(Xte6[sel5[0]], bg6, f6)[0]))

# --------------------------------------------------------------------------- #
# Task 2.6  Calculos de apoyo para las respuestas
# --------------------------------------------------------------------------- #
# (a) cliente rechazado: efecto individual en probabilidad
i = i_bad
p_x, p0 = float(sigmoid(f_raw(Xte[i:i + 1])[0])), float(sigmoid(base))
eff_j = sigmoid(base + phi[i]) - sigmoid(base)
order = np.argsort(-np.abs(eff_j))
R["t26a"] = dict(p_x=p_x, p0=p0, sum_effects=float(eff_j.sum()), diff=p_x - p0,
                 top_effects=[(feats[j], float(eff_j[j])) for j in order[:5]])

# (b) variable con mayor diferencia de posicion entre rankings
rk_s, rk_p = rankdata(-I), rankdata(-Ip)
jdiff = int(np.argmax(np.abs(rk_s - rk_p)))
Xdf = pd.DataFrame(Xte, columns=feats)
corr_s = Xdf.corr(method="spearman")[feats[jdiff]].drop(feats[jdiff])
cs = corr_s.reindex(corr_s.abs().sort_values(ascending=False).index)[:4]
perm_tr = permutation_importance(model, Xtr, ytr, scoring="roc_auc", n_repeats=10, random_state=SEED)
phi_tr = expl.shap_values(Xtr, check_additivity=False)
R["t26b_train"] = dict(perm_train=float(perm_tr.importances_mean[jdiff]), perm_train_std=float(perm_tr.importances_std[jdiff]),
                       I_train=float(np.abs(phi_tr[:, jdiff]).mean()),
                       auc_train=float(roc_auc_score(ytr, model.predict_proba(Xtr)[:, 1])),
                       spearman_phi_x=float(spearmanr(phi[:, jdiff], Xte[:, jdiff])[0]))
R["t26b"] = dict(var=feats[jdiff], rank_shap=int(rk_s[jdiff]), rank_perm=int(rk_p[jdiff]),
                 I=float(I[jdiff]), perm=float(Ip[jdiff]), perm_std=float(perm.importances_std[jdiff]),
                 top_corr=[(k, float(v)) for k, v in cs.items()],
                 ranks={feats[j]: (int(rk_s[j]), int(rk_p[j])) for j in range(d)})

# (d) costos
t100 = []
for _ in range(300):
    t0 = time.perf_counter()
    f_raw(Xte[:100])
    t100.append(time.perf_counter() - t0)
t100 = float(np.median(t100))
year = 365.25 * 24 * 3600
R["t26d"] = dict(t_pred_100=t100, t_exact_20=t100 * 2 ** 20, t_exact_20_h=t100 * 2 ** 20 / 3600,
                 t_kernel_inst_100=rows[-1]["t_kernel_inst"],
                 evals_kernel=2000 * 100, evals_exact=2 ** 20 * 100,
                 years_60=2 ** 60 * 1e-6 / year)

json.dump(R, open(DIR / "resultados/task2.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False))
