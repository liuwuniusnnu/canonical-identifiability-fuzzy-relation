import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np
import torch

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parent
LATEX_DIR = ROOT.parent / 'latex_source'
FIGDIR = LATEX_DIR / 'figures'
FIGDIR.mkdir(parents=True, exist_ok=True)
PARTS = ROOT / 'results_parts'
PARTS.mkdir(parents=True, exist_ok=True)
RESULTS = LATEX_DIR / 'results.json'

LR = 0.26
EPOCHS = 500
TAU0 = 10.0
TAU1 = 34.0


def maxmin_compose(X, R):
    return torch.max(torch.minimum(X.unsqueeze(2), R.unsqueeze(0)), dim=1).values


def smooth_min(a, b, tau):
    return -(1.0 / tau) * torch.log(torch.exp(-tau * a) + torch.exp(-tau * b))


def smooth_compose(X, R, tau=18.0):
    Z = smooth_min(X.unsqueeze(2), R.unsqueeze(0), tau)
    return (1.0 / tau) * torch.logsumexp(tau * Z, dim=1)


def make_measurement_operator(d, p, seed=0):
    rng = np.random.default_rng(seed)
    B = rng.normal(0.0, 0.22, size=(d, p)).astype(np.float32)
    for i in range(d):
        B[i, i % p] += 1.7
        B[i, (2 * i + 1) % p] += 0.8
        B[i, (3 * i + 2) % p] += 0.35
    while np.linalg.matrix_rank(B) < d:
        B += rng.normal(0.0, 0.03, size=B.shape).astype(np.float32)
    A = (B.T @ np.linalg.inv(B @ B.T)).astype(np.float32)
    return A, B


def make_relation(d, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.uniform(0.10, 0.90, size=(d, d)).astype(np.float32)
    for i in range(d):
        M[i, i] = rng.uniform(0.72, 0.95)
        if i + 1 < d:
            M[i, i + 1] = rng.uniform(0.28, 0.58)
        if i > 0:
            M[i, i - 1] = rng.uniform(0.12, 0.35)
    return M


def make_witness_states(M, seed=0, margin=0.06):
    rng = np.random.default_rng(seed)
    d = M.shape[0]
    states = [np.eye(d, dtype=np.float32)[i] for i in range(d)]
    pairs = [(i, j) for i in range(d) for j in range(d)]
    if d > 20:
        keep = min(len(pairs), 12 * d)
        ids = rng.choice(len(pairs), size=keep, replace=False)
        pairs = [pairs[k] for k in ids]
    for i, j in pairs:
        m = float(M[i, j])
        x = np.zeros(d, dtype=np.float32)
        x[i] = min(max(m + margin, 0.0), 0.98) + (1.0 - min(max(m + margin, 0.0), 0.98)) * rng.uniform()
        upper = max(m - margin, 0.0)
        for k in range(d):
            if k != i:
                x[k] = rng.uniform(0.0, upper)
        states.append(x)
    return np.array(states, dtype=np.float32)


def make_dataset(d=5, p=12, n_random=120, noise=0.05, seed=0):
    M = make_relation(d, seed)
    A_true, B_true = make_measurement_operator(d, p, seed + 1)
    base = make_witness_states(M, seed + 2)
    rng = np.random.default_rng(seed + 3)
    X_rand = rng.uniform(0.0, 1.0, size=(n_random, d)).astype(np.float32)
    X = np.vstack([base, X_rand]).astype(np.float32)
    Y = maxmin_compose(torch.tensor(X), torch.tensor(M)).numpy().astype(np.float32)
    S = (X @ B_true).astype(np.float32)
    T = (Y @ B_true).astype(np.float32)
    if noise > 0:
        S += rng.normal(0.0, noise, size=S.shape).astype(np.float32)
        T += rng.normal(0.0, noise, size=T.shape).astype(np.float32)
    return {
        'X': X,
        'Y': Y,
        'S': S,
        'T': T,
        'M': M,
        'A_true': A_true,
        'B_true': B_true,
        'anchors_q': B_true.copy(),
        'anchors_e': np.eye(d, dtype=np.float32),
    }


class Model(torch.nn.Module):
    def __init__(self, p, d, A0=None, M0=None, B0=None):
        super().__init__()
        if A0 is None:
            A0 = np.full((p, d), 0.5, dtype=np.float32)
        if M0 is None:
            M0 = np.full((d, d), 0.5, dtype=np.float32)
        if B0 is None:
            B0 = np.full((d, p), 0.5, dtype=np.float32)
        self.A = torch.nn.Parameter(torch.tensor(A0, dtype=torch.float32))
        eps = 1e-4
        U0 = np.log(np.clip(M0, eps, 1 - eps) / np.clip(1 - M0, eps, 1 - eps)).astype(np.float32)
        self.rawM = torch.nn.Parameter(torch.tensor(U0, dtype=torch.float32))
        self.B = torch.nn.Parameter(torch.tensor(B0, dtype=torch.float32))

    def M(self):
        return torch.sigmoid(self.rawM)

    def encode(self, S):
        return S @ self.A

    def decode(self, X):
        return X @ self.B

    def forward(self, S, T, tau=18.0):
        X_hat = self.encode(S)
        Y_hat = self.encode(T)
        X_box = torch.clamp(X_hat, 0.0, 1.0)
        Y_box = torch.clamp(Y_hat, 0.0, 1.0)
        Y_pred = smooth_compose(X_box, self.M(), tau=tau)
        S_rec = self.decode(X_hat)
        T_rec = self.decode(Y_hat)
        T_pred = self.decode(Y_pred)
        return X_hat, Y_hat, X_box, Y_box, Y_pred, S_rec, T_rec, T_pred


def anchor_initialization(data, seed=0, use_anchor=True):
    p = data['S'].shape[1]
    d = data['M'].shape[0]
    if use_anchor:
        return (
            np.full((p, d), 0.5, dtype=np.float32),
            np.full((d, d), 0.5, dtype=np.float32),
            np.full((d, p), 0.5, dtype=np.float32),
        )
    rng = np.random.default_rng(seed + 12345)
    A0 = rng.normal(0.0, 0.18, size=(p, d)).astype(np.float32)
    col_bias = np.linspace(-0.22, 0.26, d, dtype=np.float32)
    A0 += col_bias.reshape(1, d)
    M0 = rng.uniform(0.15, 0.85, size=(d, d)).astype(np.float32)
    for i in range(d):
        M0[i, i] = rng.uniform(0.55, 0.80)
    B0 = rng.normal(0.0, 0.18, size=(d, p)).astype(np.float32)
    row_bias = np.linspace(0.25, -0.20, d, dtype=np.float32)
    B0 += row_bias.reshape(d, 1)
    return A0, M0, B0


def permutation_error(M_hat, M_true):
    d = M_true.shape[0]
    if d > 8:
        return float('nan')
    best = float('inf')
    for perm in itertools.permutations(range(d)):
        P = np.eye(d)[list(perm)]
        cand = P.T @ M_hat @ P
        err = np.linalg.norm(cand - M_true) / np.linalg.norm(M_true)
        best = min(best, float(err))
    return best


def train_model(data, epochs=EPOCHS, seed=0, use_anchor=True, record_curves=False, record_matrix=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    S = torch.tensor(data['S'], dtype=torch.float32)
    T = torch.tensor(data['T'], dtype=torch.float32)
    Q = torch.tensor(data['anchors_q'], dtype=torch.float32)
    E = torch.tensor(data['anchors_e'], dtype=torch.float32)
    M_true = data['M']
    A_true = data['A_true']
    B_true = data['B_true']
    p = S.shape[1]
    d = M_true.shape[0]

    A0, M0, B0 = anchor_initialization(data, seed=seed, use_anchor=use_anchor)
    model = Model(p, d, A0=A0, M0=M0, B0=B0)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    hist = {
        'operator_error': [], 'prediction_mse': [], 'A_error': [], 'B_error': [], 'mean_entry_abs_error': [],
        'A_entries': {}, 'B_entries': {}, 'tracked_entries': {}
    }
    if record_curves and d <= 8:
        hist['best_permutation_error'] = []
    M_history = [] if record_matrix else None

    A_entries = [(0, 0), (min(p - 1, 2), min(d - 1, 1)), (min(p - 1, 4), min(d - 1, 3))]
    B_entries = [(0, 0), (min(d - 1, 1), min(p - 1, 3)), (min(d - 1, 3), min(p - 1, 5))]

    def choose_spread_entries(M_true, k=4):
        flat = [((i, j), float(M_true[i, j])) for i in range(M_true.shape[0]) for j in range(M_true.shape[1])]
        flat.sort(key=lambda x: x[1])
        if len(flat) <= k:
            return [idx for idx, _ in flat]
        pos = np.linspace(0, len(flat) - 1, k)
        used = set()
        chosen = []
        for pp in pos:
            cand = int(round(pp))
            while cand in used and cand + 1 < len(flat):
                cand += 1
            used.add(cand)
            chosen.append(flat[cand][0])
        return chosen

    tracked_entries = choose_spread_entries(M_true, k=min(4, d * d))
    for idx in tracked_entries:
        hist['tracked_entries'][str(idx)] = []
    for idx in A_entries:
        hist['A_entries'][str(idx)] = []
    for idx in B_entries:
        hist['B_entries'][str(idx)] = []

    for ep in range(epochs):
        tau = TAU0 + (TAU1 - TAU0) * ep / max(1, epochs - 1)
        X_hat, Y_hat, X_box, Y_box, Y_pred, S_rec, T_rec, T_pred = model(S, T, tau=tau)
        loss_rel = ((Y_pred - Y_box) ** 2).mean()
        loss_rec = ((S_rec - S) ** 2).mean() + ((T_rec - T) ** 2).mean()
        loss_pred = ((T_pred - T) ** 2).mean()
        loss_box = (
            torch.relu(-X_hat).pow(2).mean() + torch.relu(X_hat - 1.0).pow(2).mean() +
            torch.relu(-Y_hat).pow(2).mean() + torch.relu(Y_hat - 1.0).pow(2).mean()
        )
        loss_reg = 2e-4 * (model.A.pow(2).mean() + model.B.pow(2).mean() + model.M().pow(2).mean())
        loss = loss_rel + 0.28 * loss_rec + 0.90 * loss_pred + 0.50 * loss_box + loss_reg
        if use_anchor:
            loss = loss + 3.2 * ((model.encode(Q) - E) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
        opt.step()

        with torch.no_grad():
            _, _, _, _, _, _, _, T_pred_all = model(S, T, tau=TAU1)
            M_hat = model.M().cpu().numpy()
            A_hat = model.A.detach().cpu().numpy()
            B_hat = model.B.detach().cpu().numpy()
            hist['operator_error'].append(float(np.linalg.norm(M_hat - M_true) / np.linalg.norm(M_true)))
            hist['prediction_mse'].append(float(((T_pred_all - T) ** 2).mean().item()))
            hist['A_error'].append(float(np.linalg.norm(A_hat - A_true) / np.linalg.norm(A_true)))
            hist['B_error'].append(float(np.linalg.norm(B_hat - B_true) / np.linalg.norm(B_true)))
            hist['mean_entry_abs_error'].append(float(np.mean(np.abs(M_hat - M_true))))
            for idx in tracked_entries:
                hist['tracked_entries'][str(idx)].append(float(M_hat[idx[0], idx[1]]))
            for idx in A_entries:
                hist['A_entries'][str(idx)].append(float(A_hat[idx[0], idx[1]]))
            for idx in B_entries:
                hist['B_entries'][str(idx)].append(float(B_hat[idx[0], idx[1]]))
            if record_curves and d <= 8:
                hist['best_permutation_error'].append(permutation_error(M_hat, M_true))
            if record_matrix:
                M_history.append(M_hat.copy())

    hist['tracked_entries_true'] = {str(idx): float(M_true[idx[0], idx[1]]) for idx in tracked_entries}
    hist['A_entries_true'] = {str(idx): float(A_true[idx[0], idx[1]]) for idx in A_entries}
    hist['B_entries_true'] = {str(idx): float(B_true[idx[0], idx[1]]) for idx in B_entries}
    if record_matrix:
        hist['M_history'] = np.stack(M_history, axis=0).tolist()
    return hist


def run_replicates(builder, epochs=EPOCHS, seeds=(0, 1, 2), use_anchor=True):
    errs, preds, entry_errs = [], [], []
    for sd in seeds:
        data = builder(sd)
        hist = train_model(data, epochs=epochs, seed=sd, use_anchor=use_anchor, record_curves=False)
        errs.append(hist['operator_error'][-1])
        preds.append(hist['prediction_mse'][-1])
        entry_errs.append(hist['mean_entry_abs_error'][-1])
    return {
        'mean_error': float(np.mean(errs)),
        'std_error': float(np.std(errs)),
        'mean_pred_mse': float(np.mean(preds)),
        'std_pred_mse': float(np.std(preds)),
        'mean_entry_abs_error': float(np.mean(entry_errs)),
        'std_entry_abs_error': float(np.std(entry_errs)),
        'errors': [float(x) for x in errs],
        'preds': [float(x) for x in preds],
        'entry_errors': [float(x) for x in entry_errs],
    }


def draw_framework():
    fig, ax = plt.subplots(figsize=(12.4, 6.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    def rbox(x, y, w, h, txt, fc='#ffffff', ec='#4a4a4a', fontsize=11.0, lw=1.4, rounding=0.022):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.012,rounding_size={rounding}",
            linewidth=lw, edgecolor=ec, facecolor=fc
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=fontsize)
        return patch

    def arrow(x0, y0, x1, y1, text=None, dashed=False, dy=0.018, fs=10.0):
        ax.annotate(
            '', xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle='->', lw=1.6 if not dashed else 1.3, color='#4a4a4a',
                            linestyle='--' if dashed else '-')
        )
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha='center', va='center', fontsize=fs)

    # stage bands
    ax.text(0.105, 0.92, 'External-measurement layer', ha='center', va='center', fontsize=12.4, fontweight='bold')
    ax.text(0.44, 0.92, 'Latent fuzzy layer', ha='center', va='center', fontsize=12.4, fontweight='bold')
    ax.text(0.77, 0.92, 'Prediction and calibration', ha='center', va='center', fontsize=12.4, fontweight='bold')
    ax.plot([0.03, 0.19], [0.895, 0.895], lw=1.3, color='#6d6d6d')
    ax.plot([0.27, 0.61], [0.895, 0.895], lw=1.3, color='#6d6d6d')
    ax.plot([0.66, 0.97], [0.895, 0.895], lw=1.3, color='#6d6d6d')

    # top pipeline
    rbox(0.03, 0.64, 0.16, 0.15, 'Front measurement\n$\mathbf{s}_n \in \mathbb{R}^p$', fc='#f6f7f8')
    rbox(0.24, 0.64, 0.12, 0.15, 'Linear encoder\n$A$', fc='#e8f1fb')
    rbox(0.40, 0.64, 0.16, 0.15, 'Encoded state\n$\hat{x}_n=s_nA\in[0,1]^d$', fc='#edf6ea')
    rbox(0.60, 0.64, 0.13, 0.15, 'Hidden relation\n$M$', fc='#fbf1e3')
    rbox(0.77, 0.64, 0.18, 0.15, 'Predicted latent output\n$\widetilde{y}_n=(s_nA)\circ M$', fc='#edf6ea')

    # lower boxes
    rbox(0.03, 0.22, 0.16, 0.15, 'Rear measurement\n$\mathbf{t}_n \in \mathbb{R}^p$', fc='#f6f7f8')
    rbox(0.32, 0.18, 0.18, 0.16, 'Anchor responses\n$q^{(k)}A=e_k$', fc='#f6ecfa')
    rbox(0.60, 0.22, 0.13, 0.15, 'Linear decoder\n$B$', fc='#e8f1fb')
    rbox(0.77, 0.22, 0.18, 0.15, 'Predicted rear output\n$\widetilde{t}_n=((s_nA)\circ M)B$', fc='#f6f7f8', fontsize=10.3)

    # arrows
    arrow(0.19, 0.715, 0.24, 0.715)
    arrow(0.36, 0.715, 0.40, 0.715)
    arrow(0.56, 0.715, 0.60, 0.715)
    arrow(0.73, 0.715, 0.77, 0.715)
    arrow(0.665, 0.64, 0.665, 0.37, text='decode', dy=0.03)
    arrow(0.73, 0.295, 0.77, 0.295)
    arrow(0.19, 0.295, 0.77, 0.665, dashed=True, text='latent consistency target: $t_nA$', dy=0.035)

    # explanatory callouts
    rbox(0.24, 0.46, 0.32, 0.09, r'Encoded relation equation: $(s_nA)\circ M \approx t_nA$', fc='#ffffff', ec='#8a8a8a', fontsize=10.8, lw=1.1, rounding=0.018)
    rbox(0.59, 0.46, 0.34, 0.09, r'Observable prediction equation: $((s_nA)\circ M)B \approx t_n$', fc='#ffffff', ec='#8a8a8a', fontsize=10.6, lw=1.1, rounding=0.018)

    # subtle legend
    handles = [
        Line2D([0], [0], color='#4a4a4a', lw=1.6, label='data / model flow'),
        Line2D([0], [0], color='#4a4a4a', lw=1.3, linestyle='--', label='target alignment'),
    ]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False, fontsize=9.8)

    fig.tight_layout(pad=0.2)
    fig.savefig(FIGDIR / 'framework.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_smooth_fit_effect():
    M = torch.tensor(make_relation(5, seed=321), dtype=torch.float32)
    x = torch.tensor([[0.14, 0.33, 0.57, 0.76, 0.91]], dtype=torch.float32)
    exact = maxmin_compose(x, M).numpy()[0]
    taus = [6.0, 12.0, 24.0, 36.0]
    curves = {tau: smooth_compose(x, M, tau=tau).detach().numpy()[0] for tau in taus}
    coords = np.arange(1, M.shape[1] + 1)

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.plot(coords, exact, marker='o', linewidth=2.2, color='red', label='Exact max-min output')
    for tau, style in zip(taus, ['--', '-.', ':', '-']):
        ax.plot(coords, curves[tau], marker='o', linewidth=1.8, linestyle=style, label=fr'Smooth output ($\tau={int(tau)}$)')
    ax.set_xlabel('Output coordinate')
    ax.set_ylabel('Output value')
    ax.set_xticks(coords)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, ncol=2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / 'smooth_maxmin_fit_effect.pdf')
    plt.close(fig)


def plot_anchor_effect(results, hist_yes, hist_no):
    epochs = np.arange(1, len(hist_yes['operator_error_mean']) + 1)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(epochs, hist_yes['operator_error_mean'], linewidth=2.1, label='Canonical error (with anchors)')
    ax.fill_between(epochs,
                    np.maximum(0.0, np.array(hist_yes['operator_error_mean']) - np.array(hist_yes['operator_error_std'])),
                    np.array(hist_yes['operator_error_mean']) + np.array(hist_yes['operator_error_std']),
                    alpha=0.18)
    ax.plot(epochs, hist_no['operator_error_mean'], linewidth=2.0, linestyle='-', label='Canonical error (without anchors)')
    ax.fill_between(epochs,
                    np.maximum(0.0, np.array(hist_no['operator_error_mean']) - np.array(hist_no['operator_error_std'])),
                    np.array(hist_no['operator_error_mean']) + np.array(hist_no['operator_error_std']),
                    alpha=0.18)
    if 'best_permutation_error_mean' in hist_no:
        ax.plot(epochs, hist_no['best_permutation_error_mean'], linewidth=2.0, linestyle='--', label='Best-permutation error (without anchors)')
        ax.fill_between(epochs,
                        np.maximum(0.0, np.array(hist_no['best_permutation_error_mean']) - np.array(hist_no['best_permutation_error_std'])),
                        np.array(hist_no['best_permutation_error_mean']) + np.array(hist_no['best_permutation_error_std']),
                        alpha=0.14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Relative relation-matrix error')
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / 'anchor_learning_curve.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.7, 4.0))
    labels = ['Without anchors', 'With anchors']
    means = [results['anchor_without']['mean_error'], results['anchor_with']['mean_error']]
    errs = [results['anchor_without']['std_error'], results['anchor_with']['std_error']]
    xpos = np.arange(len(labels))
    bars = ax.bar(xpos, means, color=['#9db7d5', '#89c48a'], edgecolor='black', linewidth=0.8)
    ax.errorbar(xpos, means, yerr=errs, fmt='none', ecolor='black', elinewidth=1.2, capsize=4, capthick=1.2, zorder=3)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Final relative relation-matrix error')
    ax.set_ylim(0.0, max(1.0, max(means) + 0.12))
    ax.grid(True, axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGDIR / 'anchor_bar.pdf')
    plt.close(fig)


def plot_entry_tracking(hist, true_M):
    epochs = np.arange(1, len(hist['operator_error']) + 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for label, vals in hist['tracked_entries'].items():
        i, j = [int(x.strip()) for x in label.strip('()').split(',')]
        ax.plot(epochs, vals, linewidth=1.9, label=rf'Recovered $M_{{{i+1},{j+1}}}$')
        ax.axhline(hist['tracked_entries_true'][label], color='red', linestyle='--', linewidth=1.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Entry value')
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / 'relation_entry_tracking.pdf')
    plt.close(fig)

def plot_full_matrix_history(M_history, true_M):
    d = true_M.shape[0]
    epochs = np.arange(1, M_history.shape[0] + 1)
    fig, axes = plt.subplots(d, d, figsize=(2.25 * d, 2.0 * d), sharex=True, sharey=True)
    if d == 1:
        axes = np.array([[axes]])
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            ax.plot(epochs, M_history[:, i, j], linewidth=1.0)
            ax.axhline(float(true_M[i, j]), color='red', linestyle='--', linewidth=0.9)
            ax.set_title(f'({i+1},{j+1})', fontsize=7)
            ax.grid(True, alpha=0.18)
            ax.set_ylim(0.0, 1.0)
    axes[-1, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Value')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'relation_matrix_grid.pdf')
    plt.close(fig)


def plot_linear_operator_learning(hist):
    epochs = np.arange(1, len(hist['A_error']) + 1)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(epochs, hist['A_error'], label='Encoder error $\\|A-A_\\star\\|/\\|A_\\star\\|$')
    ax.plot(epochs, hist['B_error'], label='Decoder error $\\|B-B_\\star\\|/\\|B_\\star\\|$')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Relative linear-operator error')
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / 'linear_operator_learning.pdf')
    plt.close(fig)


def plot_linear_selected_entries(hist):
    epochs = np.arange(1, len(hist['A_error']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for label, vals in hist['A_entries'].items():
        axes[0].plot(epochs, vals, label=rf'$A_{{{label[1:-1]}}}$')
        axes[0].axhline(hist['A_entries_true'][label], linestyle='--', linewidth=1.0)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Entry value')
    axes[0].set_title('Representative entries of $A$')
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    for label, vals in hist['B_entries'].items():
        axes[1].plot(epochs, vals, label=rf'$B_{{{label[1:-1]}}}$')
        axes[1].axhline(hist['B_entries_true'][label], linestyle='--', linewidth=1.0)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Entry value')
    axes[1].set_title('Representative entries of $B$')
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / 'linear_operator_selected_entries.pdf')
    plt.close(fig)


def band_plot(x, y, yerr, xlabel, ylabel, fname, xticks=None):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    yerr = np.array(yerr, dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    ax.plot(x, y, marker='o', linewidth=1.8)
    ax.fill_between(x, np.maximum(0.0, y - yerr), y + yerr, alpha=0.22)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0.0)
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(v) for v in xticks], rotation=0)
    ax.grid(True, alpha=0.28)
    fig.tight_layout()
    fig.savefig(FIGDIR / fname)
    plt.close(fig)


def sample_dual_plot(x, op_mean, op_std, mse_mean, mse_std):
    x = np.array(x, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    axes[0].plot(x, op_mean, marker='o', linewidth=1.8)
    axes[0].fill_between(x, np.maximum(0.0, np.array(op_mean) - np.array(op_std)), np.array(op_mean) + np.array(op_std), alpha=0.22)
    axes[0].set_xlabel('Additional paired external measurements')
    axes[0].set_ylabel('Final operator error')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(int(v)) for v in x], rotation=45)
    axes[0].grid(True, alpha=0.28)
    axes[1].plot(x, mse_mean, marker='o', linewidth=1.8)
    axes[1].fill_between(x, np.maximum(0.0, np.array(mse_mean) - np.array(mse_std)), np.array(mse_mean) + np.array(mse_std), alpha=0.22)
    axes[1].set_xlabel('Additional paired external measurements')
    axes[1].set_ylabel('Prediction MSE')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(int(v)) for v in x], rotation=45)
    axes[1].grid(True, alpha=0.28)
    fig.tight_layout()
    fig.savefig(FIGDIR / 'sample_effect_dual.pdf')
    plt.close(fig)


def sample_efficiency_plot(x, op_success, mse_success):
    x = np.array(x, dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    ax.plot(x, op_success, marker='o', linewidth=1.8, label='Success rate for operator error')
    ax.plot(x, mse_success, marker='s', linewidth=1.8, linestyle='--', label='Success rate for prediction MSE')
    ax.set_xlabel('Additional paired external measurements')
    ax.set_ylabel('Success probability')
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x], rotation=45)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / 'sample_efficiency.pdf')
    plt.close(fig)

def sample_dimension_plot(sample_points, dim_results):
    x = np.array(sample_points, dtype=float)
    dims = sorted(dim_results)
    y = np.array(dims, dtype=float)
    X, Y = np.meshgrid(x, y)
    Z = np.array([dim_results[d]['mean_error'] for d in dims], dtype=float)

    fig = plt.figure(figsize=(8.2, 5.0))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0.35, edgecolor='k', antialiased=True, alpha=0.95)
    for j, d in enumerate(dims):
        ax.plot(x, np.full_like(x, d, dtype=float), Z[j], linewidth=1.2)
    ax.set_xlabel('Additional paired external measurements', labelpad=9)
    ax.set_ylabel('Latent dimension $d$', labelpad=8)
    ax.set_zlabel('Mean absolute entrywise error', labelpad=8)
    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.view_init(elev=26, azim=-128)
    fig.colorbar(surf, shrink=0.72, pad=0.10, aspect=18)
    fig.tight_layout()
    fig.savefig(FIGDIR / 'sample_dimension_surface.pdf', bbox_inches='tight')
    plt.close(fig)


def save_part(name, data):
    (PARTS / f'{name}.json').write_text(json.dumps(data, indent=2))


def load_results():
    out = {}
    for p in sorted(PARTS.glob('*.json')):
        out[p.stem] = json.loads(p.read_text())
    return out


def stage_representative():
    draw_framework()
    plot_smooth_fit_effect()
    rep_data = make_dataset(d=5, p=18, n_random=180, noise=0.10, seed=40)
    hist_yes = train_model(rep_data, epochs=EPOCHS, seed=0, use_anchor=True, record_curves=True, record_matrix=True)
    hist_no = train_model(rep_data, epochs=EPOCHS, seed=0, use_anchor=False, record_curves=True)
    plot_entry_tracking(hist_yes, rep_data['M'])
    plot_full_matrix_history(np.array(hist_yes['M_history']), rep_data['M'])
    plot_linear_operator_learning(hist_yes)
    plot_linear_selected_entries(hist_yes)
    save_part('representative', {
        'rep_true_M': rep_data['M'].tolist(),
        'with_anchor_hist': {k: v for k, v in hist_yes.items() if k != 'M_history'},
        'without_anchor_hist': hist_no,
    })
    return load_results()['representative']




def aggregate_curve_histories(builder, seeds=(0,1,2,3,4), use_anchor=True):
    hists = []
    for sd in seeds:
        data = builder(sd)
        hist = train_model(data, epochs=EPOCHS, seed=sd, use_anchor=use_anchor, record_curves=True)
        hists.append(hist)
    out = {}
    for key in ['operator_error', 'prediction_mse', 'A_error', 'B_error', 'mean_entry_abs_error']:
        arr = np.array([h[key] for h in hists], dtype=float)
        out[f'{key}_mean'] = arr.mean(axis=0).tolist()
        out[f'{key}_std'] = arr.std(axis=0).tolist()
    if all('best_permutation_error' in h for h in hists):
        arr = np.array([h['best_permutation_error'] for h in hists], dtype=float)
        out['best_permutation_error_mean'] = arr.mean(axis=0).tolist()
        out['best_permutation_error_std'] = arr.std(axis=0).tolist()
    return out
def stage_anchor():
    curve_seeds = (0, 1, 2, 3, 4)
    stats_seeds = (0, 1, 2, 3, 4)
    hist_yes = aggregate_curve_histories(lambda sd: make_dataset(d=5, p=18, n_random=180, noise=0.10, seed=40 + sd), seeds=curve_seeds, use_anchor=True)
    hist_no = aggregate_curve_histories(lambda sd: make_dataset(d=5, p=18, n_random=180, noise=0.10, seed=60 + sd), seeds=curve_seeds, use_anchor=False)
    results = {}
    results['anchor_without'] = run_replicates(lambda sd: make_dataset(d=5, p=18, n_random=180, noise=0.10, seed=80 + sd), epochs=EPOCHS, seeds=stats_seeds, use_anchor=False)
    results['anchor_with'] = run_replicates(lambda sd: make_dataset(d=5, p=18, n_random=180, noise=0.10, seed=120 + sd), epochs=EPOCHS, seeds=stats_seeds, use_anchor=True)
    results['anchor_curve_with'] = hist_yes
    results['anchor_curve_without'] = hist_no
    plot_anchor_effect(results, hist_yes, hist_no)
    save_part('anchor', results)
    return results


def stage_dimension():
    dim_points = [2, 3, 5, 8, 10, 15, 20, 30, 40, 50]
    dim_mean, dim_std = [], []
    for d in dim_points:
        p = max(d + 8, int(1.6 * d) + 6)
        out = run_replicates(lambda sd, d=d, p=p: make_dataset(d=d, p=p, n_random=120, noise=0.08, seed=200 + 7 * d + sd), epochs=EPOCHS, seeds=(0, 1, 2), use_anchor=True)
        dim_mean.append(out['mean_error'])
        dim_std.append(out['std_error'])
    res = {'dims': dim_points, 'mean': dim_mean, 'std': dim_std}
    save_part('dimension_sweep', res)
    band_plot(dim_points, dim_mean, dim_std, 'Latent dimension $d$', 'Relative relation-matrix error', 'dimension_effect.pdf', xticks=dim_points)
    return res


def stage_sample():
    sample_points = [20, 40, 60, 80, 100, 120, 150, 180, 210, 240, 270, 300]
    sample_mean, sample_std, mse_mean, mse_std = [], [], [], []
    for n in sample_points:
        out = run_replicates(lambda sd, n=n: make_dataset(d=5, p=24, n_random=n, noise=0.03, seed=500 + sd), epochs=EPOCHS, seeds=(0, 1, 2), use_anchor=True)
        sample_mean.append(out['mean_error'])
        sample_std.append(out['std_error'])
        mse_mean.append(out['mean_pred_mse'])
        mse_std.append(out['std_pred_mse'])
    res = {'sample_sizes': sample_points, 'mean_error': sample_mean, 'std_error': sample_std, 'mean_pred_mse': mse_mean, 'std_pred_mse': mse_std}
    save_part('sample_sweep', res)
    sample_dual_plot(sample_points, sample_mean, sample_std, mse_mean, mse_std)
    return res


def stage_sample_dimension():
    sample_points = [20, 40, 60, 80, 100, 120, 150, 180, 210, 240, 270, 300]
    dims = [2, 5, 10, 20]
    dim_results = {}
    for d in dims:
        p = max(18, 2 * d + 8)
        mean_error, std_error = [], []
        for n in sample_points:
            out = run_replicates(lambda sd, d=d, p=p, n=n: make_dataset(d=d, p=p, n_random=n, noise=0.03, seed=1700 + 13 * d + n + sd), epochs=EPOCHS, seeds=(0, 1, 2), use_anchor=True)
            mean_error.append(out['mean_entry_abs_error'])
            std_error.append(out['std_entry_abs_error'])
        dim_results[d] = {'mean_error': mean_error, 'std_error': std_error, 'measurement_dim': p}
    res = {'sample_sizes': sample_points, 'dimension_curves': dim_results, 'metric': 'mean absolute entrywise error'}
    save_part('sample_dimension_sweep', res)
    sample_dimension_plot(sample_points, dim_results)
    return res


def stage_measurement():
    sensor_points = [10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 64, 80, 100]
    sensor_mean, sensor_std = [], []
    for p in sensor_points:
        out = run_replicates(lambda sd, p=p: make_dataset(d=5, p=p, n_random=180, noise=0.03, seed=900 + sd), epochs=EPOCHS, seeds=(0, 1, 2), use_anchor=True)
        sensor_mean.append(out['mean_error'])
        sensor_std.append(out['std_error'])
    res = {'measurement_dims': sensor_points, 'mean': sensor_mean, 'std': sensor_std}
    save_part('measurement_sweep', res)
    band_plot(sensor_points, sensor_mean, sensor_std, 'External-measurement dimension $p$', 'Relative relation-matrix error', 'measurement_effect.pdf', xticks=sensor_points)
    return res


def stage_noise():
    noise_points = [0.00, 0.03, 0.06, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    noise_mean, noise_std = [], []
    for nz in noise_points:
        out = run_replicates(lambda sd, nz=nz: make_dataset(d=5, p=24, n_random=180, noise=nz, seed=1200 + sd), epochs=EPOCHS, seeds=(0, 1, 2), use_anchor=True)
        noise_mean.append(out['mean_error'])
        noise_std.append(out['std_error'])
    res = {'noise': noise_points, 'mean': noise_mean, 'std': noise_std}
    save_part('noise_sweep', res)
    band_plot(noise_points, noise_mean, noise_std, 'Noise standard deviation', 'Relative relation-matrix error', 'noise_effect.pdf', xticks=noise_points)
    return res


def stage_merge():
    out = load_results()
    out['training_setup'] = {'initial_value': 0.5, 'epochs': EPOCHS, 'optimizer': 'Adam', 'learning_rate': LR, 'no_anchor_initialization': 'asymmetric random'}
    RESULTS.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='all', choices=['representative', 'anchor', 'dimension', 'sample', 'sample_dimension', 'measurement', 'noise', 'merge', 'all'])
    args = parser.parse_args()

    if args.stage == 'representative':
        stage_representative()
    elif args.stage == 'anchor':
        stage_anchor()
    elif args.stage == 'dimension':
        stage_dimension()
    elif args.stage == 'sample':
        stage_sample()
    elif args.stage == 'sample_dimension':
        stage_sample_dimension()
    elif args.stage == 'measurement':
        stage_measurement()
    elif args.stage == 'noise':
        stage_noise()
    elif args.stage == 'merge':
        stage_merge()
    elif args.stage == 'all':
        stage_representative()
        stage_anchor()
        stage_dimension()
        stage_sample()
        stage_sample_dimension()
        stage_measurement()
        stage_noise()
        stage_merge()


if __name__ == '__main__':
    main()
