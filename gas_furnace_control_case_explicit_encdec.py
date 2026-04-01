from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from linear_sensor_maxmin_fuzzy_relation_experiments import Model, smooth_compose, TAU0, TAU1

ROOT = Path(__file__).resolve().parent
LATEX_DIR = ROOT.parent / 'latex_source'
FIGDIR = LATEX_DIR / 'figures'
FIGDIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = LATEX_DIR / 'gas_furnace_case_metrics.json'
DATA_FILE = ROOT / 'gas_furnace_series.csv'
RAW_FILE = ROOT / 'gas_furnace_raw.txt'

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

torch.set_num_threads(2)


def ensure_datafile() -> pd.DataFrame:
    if DATA_FILE.exists():
        return pd.read_csv(DATA_FILE)
    if not RAW_FILE.exists():
        raise FileNotFoundError('Need gas_furnace_raw.txt or gas_furnace_series.csv')
    vals = [float(x) for x in RAW_FILE.read_text().strip().split()]
    arr = np.array(vals, dtype=float).reshape(-1, 2)
    df = pd.DataFrame({'gas_rate': arr[:, 0], 'co2': arr[:, 1]})
    df.to_csv(DATA_FILE, index=False)
    return df


def build_case() -> dict:
    df = ensure_datafile().copy()
    u = df['gas_rate'].to_numpy()
    y = df['co2'].to_numpy()
    n = len(df)
    rows_s = []
    rows_t = []
    meta = []
    for t in range(2, n - 1):
        du = u[t] - u[t - 1]
        dy = y[t] - y[t - 1]
        ybar_t = (y[t] + y[t - 1] + y[t - 2]) / 3.0
        ybar_tp1 = (y[t + 1] + y[t] + y[t - 1]) / 3.0
        s = [u[t], y[t], y[t - 1], du, dy, ybar_t]
        tvec = [u[t], y[t + 1], y[t], du, y[t + 1] - y[t], ybar_tp1]
        rows_s.append(s)
        rows_t.append(tvec)
        meta.append((t, y[t + 1]))
    S_raw = np.asarray(rows_s, dtype=np.float32)
    T_raw = np.asarray(rows_t, dtype=np.float32)
    idx_all = np.arange(S_raw.shape[0])
    split = int(0.7 * len(idx_all))
    train_idx = idx_all[:split]
    test_idx = idx_all[split:]

    mins = np.minimum(S_raw.min(axis=0), T_raw.min(axis=0))
    maxs = np.maximum(S_raw.max(axis=0), T_raw.max(axis=0))
    scale = np.where(maxs > mins, maxs - mins, 1.0)
    S = ((S_raw - mins) / scale).astype(np.float32)
    T = ((T_raw - mins) / scale).astype(np.float32)

    S_train = S[train_idx]
    y_train_current = S_raw[train_idx, 1]
    dy_train = S_raw[train_idx, 4]
    q20, q80 = np.quantile(y_train_current, [0.2, 0.8])
    med = np.median(y_train_current)
    low_anchor = S_train[y_train_current <= q20].mean(axis=0)
    high_anchor = S_train[y_train_current >= q80].mean(axis=0)
    nominal_mask = (np.abs(y_train_current - med) <= np.quantile(np.abs(y_train_current - med), 0.25)) & (
        np.abs(dy_train) <= np.quantile(np.abs(dy_train), 0.4)
    )
    nominal_anchor = S_train[nominal_mask].mean(axis=0)
    anchors_q = np.vstack([low_anchor, nominal_anchor, high_anchor]).astype(np.float32)
    anchors_e = np.eye(3, dtype=np.float32)

    return {
        'df': df,
        'S_raw': S_raw,
        'T_raw': T_raw,
        'S': S,
        'T': T,
        'train_idx': train_idx,
        'test_idx': test_idx,
        'time_index': np.array([m[0] for m in meta]),
        'y_next_true': np.array([m[1] for m in meta], dtype=np.float32),
        'anchors_q': anchors_q,
        'anchors_e': anchors_e,
        'feature_mins': mins.astype(np.float32),
        'feature_scale': scale.astype(np.float32),
        'y_next_min': mins[1],
        'y_next_scale': scale[1],
    }


def train_case(data: dict, d: int = 3, epochs: int = 800, seed: int = 2) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    S = torch.tensor(data['S'], dtype=torch.float32)
    T = torch.tensor(data['T'], dtype=torch.float32)
    Q = torch.tensor(data['anchors_q'], dtype=torch.float32)
    E = torch.tensor(data['anchors_e'], dtype=torch.float32)
    p = S.shape[1]

    rng = np.random.default_rng(seed)
    A0 = rng.normal(0.0, 0.16, size=(p, d)).astype(np.float32)
    B0 = rng.normal(0.0, 0.16, size=(d, p)).astype(np.float32)
    M0 = np.full((d, d), 0.45, dtype=np.float32)
    for i in range(d):
        M0[i, i] = 0.72
    model = Model(p, d, A0=A0, M0=M0, B0=B0)
    opt = torch.optim.Adam(model.parameters(), lr=0.035)

    losses = []
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
        loss_anchor = ((model.encode(Q) - E) ** 2).mean()
        loss_reg = 3e-4 * (model.A.pow(2).mean() + model.B.pow(2).mean() + model.M().pow(2).mean())
        # mild diagonal encouragement for interpretable regimes
        Mcur = model.M()
        offdiag = Mcur - torch.diag(torch.diag(Mcur))
        loss_diag = 0.05 * offdiag.pow(2).mean() - 0.03 * torch.diag(Mcur).mean()
        loss = loss_rel + 0.30 * loss_rec + 0.95 * loss_pred + 0.40 * loss_box + 2.8 * loss_anchor + loss_reg + loss_diag
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
        opt.step()
        losses.append(float(loss.item()))

    with torch.no_grad():
        X_hat, Y_hat, X_box, Y_box, Y_pred, S_rec, T_rec, T_pred = model(S, T, tau=TAU1)
        out = {
            'A': model.A.detach().cpu().numpy(),
            'B': model.B.detach().cpu().numpy(),
            'M': model.M().detach().cpu().numpy(),
            'X_current': X_box.detach().cpu().numpy(),
            'Y_pred_latent': Y_pred.detach().cpu().numpy(),
            'T_pred': T_pred.detach().cpu().numpy(),
            'S_rec': S_rec.detach().cpu().numpy(),
            'losses': losses,
        }
    return out


def denorm_feature(vals, mins, scale, idx):
    return vals[:, idx] * scale[idx] + mins[idx]


def summarize(data: dict, fit: dict) -> dict:
    mins = data['feature_mins']
    scale = data['feature_scale']
    test_idx = data['test_idx']
    train_idx = data['train_idx']

    y_true_all = data['T_raw'][:, 1]
    y_pred_all = denorm_feature(fit['T_pred'], mins, scale, 1)
    y_true_test = y_true_all[test_idx]
    y_pred_test = y_pred_all[test_idx]
    y_true_train = y_true_all[train_idx]
    y_pred_train = y_pred_all[train_idx]

    train_pred_mse = float(np.mean((fit['T_pred'][train_idx] - data['T'][train_idx]) ** 2))
    test_pred_mse = float(np.mean((fit['T_pred'][test_idx] - data['T'][test_idx]) ** 2))
    train_rec_mse = float(np.mean((fit['S_rec'][train_idx] - data['S'][train_idx]) ** 2))
    test_rec_mse = float(np.mean((fit['S_rec'][test_idx] - data['S'][test_idx]) ** 2))

    co2_rmse_train = float(np.sqrt(np.mean((y_pred_train - y_true_train) ** 2)))
    co2_rmse_test = float(np.sqrt(np.mean((y_pred_test - y_true_test) ** 2)))
    co2_mae_train = float(np.mean(np.abs(y_pred_train - y_true_train)))
    co2_mae_test = float(np.mean(np.abs(y_pred_test - y_true_test)))

    # decision policy
    lower, upper = 52.5, 54.5
    current_lat = fit['X_current'][test_idx]
    pred_lat = fit['Y_pred_latent'][test_idx]
    actions = []
    for yp, lat, predlat in zip(y_pred_test, current_lat, pred_lat):
        if (yp > upper) or (predlat[2] >= predlat[0] and predlat[2] >= predlat[1]):
            actions.append('Trim gas down')
        elif (yp < lower) or (predlat[0] >= predlat[1] and predlat[0] >= predlat[2]):
            actions.append('Trim gas up')
        else:
            actions.append('Hold')
    action_counts = {k: actions.count(k) for k in ['Trim gas down', 'Hold', 'Trim gas up']}

    return {
        'A_hat': fit['A'].tolist(),
        'B_hat': fit['B'].tolist(),
        'train_pred_mse_scaled': train_pred_mse,
        'test_pred_mse_scaled': test_pred_mse,
        'train_rec_mse_scaled': train_rec_mse,
        'test_rec_mse_scaled': test_rec_mse,
        'co2_rmse_train': co2_rmse_train,
        'co2_rmse_test': co2_rmse_test,
        'co2_mae_train': co2_mae_train,
        'co2_mae_test': co2_mae_test,
        'M_hat': fit['M'].tolist(),
        'action_counts': action_counts,
        'y_pred_test': y_pred_test.tolist(),
        'y_true_test': y_true_test.tolist(),
        'test_time_index': data['time_index'][test_idx].tolist(),
        'current_lat_test': current_lat.tolist(),
        'pred_lat_test': pred_lat.tolist(),
        'anchors_q_scaled': data['anchors_q'].tolist(),
        'anchors_q_original': (data['anchors_q'] * scale + mins).tolist(),
        'split_time_index': int(data['time_index'][test_idx[0]]),
        'raw_u': data['df']['gas_rate'].tolist(),
        'raw_y': data['df']['co2'].tolist(),
    }


def draw_figures(metrics: dict):
    t_raw = np.arange(len(metrics['raw_y']))
    split_x = metrics['split_time_index']
    # Figure 1
    fig, ax1 = plt.subplots(figsize=(7.1, 4.4))
    ax1.plot(t_raw, metrics['raw_y'], label='CO2')
    ax1.set_xlabel('Time index')
    ax1.set_ylabel('CO2')
    ax2 = ax1.twinx()
    ax2.plot(t_raw, metrics['raw_u'], linestyle='-.', label='Input gas rate')
    ax2.set_ylabel('Input gas rate')
    ax1.axvline(split_x, linestyle='--')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc='upper right')
    ax1.set_title('Gas furnace measurements with train/test split')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'gas_furnace_measurements.pdf', bbox_inches='tight')
    plt.close(fig)

    # Figure 2
    x = np.array(metrics['test_time_index'])
    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    ax.plot(x, metrics['y_true_test'], label='Observed next-step CO2')
    ax.plot(x, metrics['y_pred_test'], linestyle='--', label='Predicted next-step CO2')
    ax.axhline(52.5, linestyle=':')
    ax.axhline(54.5, linestyle=':')
    ax.set_xlabel('Time index')
    ax.set_ylabel('CO2')
    ax.set_title('One-step response prediction on the test segment')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / 'gas_furnace_prediction.pdf', bbox_inches='tight')
    plt.close(fig)

    # Figure 3 early window
    cur = np.array(metrics['current_lat_test'])
    pred = np.array(metrics['pred_lat_test'])
    window = min(18, len(x))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for k in range(cur.shape[1]):
        ax.plot(x[:window], cur[:window, k], label=f'Current latent {k+1}')
    for k in range(pred.shape[1]):
        ax.plot(x[:window], pred[:window, k], linestyle='--', label=f'Predicted next latent {k+1}')
    ax.set_xlabel('Time index')
    ax.set_ylabel('Membership degree')
    ax.set_ylim(-0.02, 1.05)
    ax.set_title('Latent regime memberships on an early test window')
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGDIR / 'gas_furnace_latent_window.pdf', bbox_inches='tight')
    plt.close(fig)

    # Figure 4 actions
    action_map = {'Trim gas up': 0, 'Hold': 1, 'Trim gas down': 2}
    inv_labels = ['Trim gas up', 'Hold', 'Trim down']
    # reconstruct actions in order from counts impossible; recompute from metrics vectors
    actions = []
    for yp, predlat in zip(metrics['y_pred_test'], metrics['pred_lat_test']):
        predlat = np.array(predlat)
        if (yp > 54.5) or (predlat[2] >= predlat[0] and predlat[2] >= predlat[1]):
            actions.append('Trim gas down')
        elif (yp < 52.5) or (predlat[0] >= predlat[1] and predlat[0] >= predlat[2]):
            actions.append('Trim gas up')
        else:
            actions.append('Hold')
    yact = np.array([action_map[a] for a in actions])
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.step(x, yact, where='mid')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(inv_labels)
    ax.set_xlabel('Time index')
    ax.set_title('Control recommendations on the test segment')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'gas_furnace_actions.pdf', bbox_inches='tight')
    plt.close(fig)




    # Figure 5 encoder and decoder heatmaps
    A = np.array(metrics['A_hat'])
    B = np.array(metrics['B_hat'])

    fig, ax = plt.subplots(figsize=(4.6, 4.8))
    im = ax.imshow(A, aspect='auto')
    ax.set_xlabel('Latent coordinate')
    ax.set_ylabel('Measurement feature')
    ax.set_xticks(range(A.shape[1]))
    ax.set_xticklabels([f'z{k+1}' for k in range(A.shape[1])])
    ax.set_yticks(range(A.shape[0]))
    ax.set_yticklabels(['u_t', 'y_t', 'y_{t-1}', 'Delta u_t', 'Delta y_t', 'ybar_t^(3)'])
    ax.set_title('Learned encoder A_hat_gas')
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f'{A[i,j]:.2f}', ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGDIR / 'gas_furnace_encoder_heatmap.pdf', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    im = ax.imshow(B, aspect='auto')
    ax.set_xlabel('Measurement feature')
    ax.set_ylabel('Latent coordinate')
    ax.set_xticks(range(B.shape[1]))
    ax.set_xticklabels(['u_t', 'y_{t+1}', 'y_t', 'Delta u_t', 'y_{t+1}-y_t', 'ybar_{t+1}^(3)'], rotation=20, ha='right')
    ax.set_yticks(range(B.shape[0]))
    ax.set_yticklabels([f'z{k+1}' for k in range(B.shape[0])])
    ax.set_title('Learned decoder B_hat_gas')
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            ax.text(j, i, f'{B[i,j]:.2f}', ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGDIR / 'gas_furnace_decoder_heatmap.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    data = build_case()
    fit = train_case(data)
    metrics = summarize(data, fit)
    draw_figures(metrics)
    OUT_JSON.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({
        'test_rmse': metrics['co2_rmse_test'],
        'test_mae': metrics['co2_mae_test'],
        'test_pred_mse': metrics['test_pred_mse_scaled'],
        'test_rec_mse': metrics['test_rec_mse_scaled'],
        'actions': metrics['action_counts'],
        'M_hat': metrics['M_hat'],
        'A_hat': metrics['A_hat'],
        'B_hat': metrics['B_hat'],
    }, indent=2))


if __name__ == '__main__':
    main()
