"""
=============================================================================
simulation_study.py

Bayesian Semiparametric Robust Estimation for Poisson Regression
Simulation Study — Extension of Jha (2025) to Count Data
+ Real‑data analysis (custom CSV) & Thall & Vail (1990) epilepsy data

References
----------
[B98]  Basu, A., Harris, I.R., Hjort, N.L. and Jones, M.C. (1998).
       Robust and efficient estimation by minimising a density power divergence.
       Biometrika, 85(3), 549–559.

[J25]  Jha, J. (2025). Bayesian semiparametric model for robust estimation.
       ISRU, Indian Statistical Institute, Kolkata.

[N00]  Neal, R.M. (2000). Markov chain sampling methods for Dirichlet process
       mixture models. JCGS 9(2), 249–265.

[TV90] Thall, P.F. and Vail, S.C. (1990). Some covariance models for
       longitudinal count data with overdispersion. Biometrics, 46, 657–671.

Outputs (written to ./results/)
--------------------------------
  simulation_table.csv              — Bias & RMSE for all methods
  fig1_scatter_fitted_epsX.png      — Scatter + fitted curves (rep dataset)
  fig2_prob_not_outlier_epsX.png    — P(Z_i=1|data) vs covariate x
  fig3_true_vs_flagged_epsX.png     — True outliers vs model-flagged
  fig4_bias_rmse.png                — Bias / RMSE across contamination levels
  fig1_real_fitted.png              — Custom real data: fitted curves
  fig2_real_prob.png                — Custom real data: P(not outlier)
  fig3_real_flagged.png             — Custom real data: flagged observations
  fig4_real_diagnostic.png          — Custom real data: diagnostic bubble plot
  real_data_results.csv             — Numerical results for custom real data
  fig1_epilepsy_fitted.png          — Epilepsy data: fitted curves
  fig2_epilepsy_prob.png            — Epilepsy data: P(not outlier)
  fig3_epilepsy_flagged.png         — Epilepsy data: flagged observations
  fig4_epilepsy_diagnostic.png      — Epilepsy data: diagnostic bubble plot
  epilepsy_results.csv              — Numerical results for epilepsy data
=============================================================================
"""

import os, sys, time, warnings, pickle
import numpy as np
import pandas as pd
from scipy.stats   import poisson, nbinom
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

# ── Output directory ──────────────────────────────────────────────────────────
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUTDIR, exist_ok=True)

# ── Global settings ───────────────────────────────────────────────────────────
MASTER_SEED = 2025
BETA_TRUE   = np.array([1.0, 0.5])      # true regression coefficients [β₀, β₁]
N           = 100                        # sample size per dataset
OUTLIER_MEAN = 25.0                      # Poisson mean for contaminating distribution
CONTAM_LEVELS = [0, 5, 10, 15]          # contamination percentages
N_REP       = 50                         # replications per contamination level
DPD_ALPHAS  = [0.10, 0.25, 0.50]        # DPD tuning parameters

# MCMC settings
N_ITER   = 12_000     # total MCMC iterations
BURN_IN  =  2_000     # burn-in (discarded)

# Prior hyperparameters
ALPHA_P   = 18        # Beta(18, 2) prior on p: mean = 0.90
BETA_P    =  2
PRIOR_VAR = 100.0     # σ²_β for N(0, σ²_β · I) prior on β
A0        =  1.0      # G0 = Gamma(a0, rate=b0): base measure for DP
B0        =  0.5      # G0 mean = a0/b0 = 2.0
ALPHA0_DP =  1.0      # DP concentration parameter

# Custom real data file (optional)
REAL_DATA_PATH = "real_data.csv"   # expected columns: x, y

print("=" * 65)
print("Bayesian Semiparametric Robust Poisson Regression")
print("Simulation Study following Jha (2025)")
print(f"n={N}  reps={N_REP}  MCMC={N_ITER}/{BURN_IN}  seed={MASTER_SEED}")
print(f"Output directory: {OUTDIR}")
print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA GENERATION  (simulation)
# ══════════════════════════════════════════════════════════════════════════════

def generate_data(n, beta_true, contamination_pct, outlier_mean, rng):
    x  = rng.uniform(-2.0, 2.0, n)
    X  = np.column_stack([np.ones(n), x])
    y  = rng.poisson(np.exp(X @ beta_true)).astype(float)
    is_out = np.zeros(n, dtype=bool)
    n_out  = int(round(n * contamination_pct / 100))
    if n_out > 0:
        idx           = rng.choice(n, n_out, replace=False)
        y[idx]        = rng.poisson(outlier_mean, size=n_out)
        is_out[idx]   = True
    return X, y, is_out


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DPD ESTIMATOR  (Basu et al. 1998)
# ══════════════════════════════════════════════════════════════════════════════

def _pmf_matrix(mu, y_max):
    k       = np.arange(0, y_max + 1, dtype=float)
    log_fac = np.zeros(y_max + 1)
    log_fac[1:] = np.cumsum(np.log(np.arange(1, y_max + 1)))
    log_pmf = (k[None, :] * np.log(np.maximum(mu, 1e-300))[:, None]
               - mu[:, None] - log_fac[None, :])
    return np.exp(log_pmf)


def dpd_objective(beta, X, y, alpha):
    n   = len(y)
    mu  = np.exp(np.clip(X @ beta, -20, 20))
    y_max = max(int(np.max(mu + 8*np.sqrt(np.maximum(mu, 1)) + 30)),
                int(y.max()) + 10)
    F   = _pmf_matrix(mu, y_max)
    A   = np.sum(F ** (1 + alpha), axis=1)
    fyi = F[np.arange(n), y.astype(int)]
    return np.mean(A - (1.0 + 1.0 / alpha) * fyi ** alpha)


def fit_mle(X, y):
    def neg_ll_g(b):
        mu  = np.exp(np.clip(X @ b, -20, 20))
        return (-np.sum(y * np.log(np.maximum(mu, 1e-300)) - mu),
                -X.T @ (y - mu))
    res = minimize(neg_ll_g, np.zeros(X.shape[1]), jac=True, method='BFGS',
                   options={'maxiter': 5000, 'gtol': 1e-10})
    return res.x


def fit_dpd(X, y, alpha, beta_init=None):
    if beta_init is None:
        beta_init = fit_mle(X, y)
    if alpha == 0.0:
        return beta_init
    res = minimize(dpd_objective, beta_init, args=(X, y, alpha),
                   method='BFGS', options={'maxiter': 5000, 'gtol': 1e-10})
    return res.x


# ══════════════════════════════════════════════════════════════════════════════
# 3.  BAYESIAN SEMIPARAMETRIC MCMC  (Jha 2025, Poisson extension)
# ══════════════════════════════════════════════════════════════════════════════

def _log_post_beta(beta, X, y, z, prior_var):
    lp  = -0.5 * np.dot(beta, beta) / prior_var
    idx = np.where(z == 1)[0]
    if len(idx) > 0:
        mu  = np.exp(np.clip(X[idx] @ beta, -20, 20))
        lp += np.sum(y[idx] * np.log(np.maximum(mu, 1e-300)) - mu)
    return lp


def run_mcmc(X, y,
             n_iter, burn_in,
             alpha_p, beta_p, prior_var,
             a0, b0, alpha0_dp,
             proposal_std=0.20):
    n, pdim  = X.shape
    y_int    = y.astype(int)
    y_fl     = y.astype(float)
    n_save   = n_iter - burn_in

    beta_cur  = fit_dpd(X, y, alpha=0.25)
    mu_init   = np.exp(np.clip(X @ beta_cur, -20, 20))
    pearson_r = (y - mu_init) / np.sqrt(np.maximum(mu_init, 1e-8))
    z_cur     = (pearson_r <= 2.576).astype(int)
    p_cur     = float(np.clip(z_cur.mean(), 0.50, 0.95))
    theta_cur = np.full(n, a0 / b0)
    idx_out0  = np.where(z_cur == 0)[0]
    if len(idx_out0) > 0:
        theta_cur[idx_out0] = np.maximum(y[idx_out0], a0 / b0)

    beta_samp = np.zeros((n_save, pdim))
    z_samp    = np.zeros((n_save, n),   dtype=np.int8)
    prop_std  = proposal_std
    acc_count = 0
    burn_acc  = 0

    for it in range(n_iter):
        mu_cur = np.exp(np.clip(X @ beta_cur, -20, 20))

        # STEP 1 — Z_i (Gibbs)
        lp_c   = (np.log(p_cur + 1e-300)
                  + poisson.logpmf(y_int, mu_cur))
        lp_o   = (np.log(1.0 - p_cur + 1e-300)
                  + poisson.logpmf(y_int, np.maximum(theta_cur, 1e-8)))
        p_cln  = np.clip(np.exp(lp_c - np.logaddexp(lp_c, lp_o)), 0.0, 1.0)
        z_cur  = (np.random.uniform(size=n) < p_cln).astype(int)

        # STEP 2 — p (Gibbs)
        s     = int(z_cur.sum())
        p_cur = np.clip(np.random.beta(alpha_p + s, beta_p + n - s),
                        0.40, 0.9999)

        # STEP 3 — β (MH)
        b_prop = beta_cur + np.random.normal(0.0, prop_std, pdim)
        log_r  = (_log_post_beta(b_prop, X, y_fl, z_cur, prior_var)
                - _log_post_beta(beta_cur, X, y_fl, z_cur, prior_var))
        if np.log(np.random.uniform()) < log_r:
            beta_cur = b_prop
            if it >= burn_in: acc_count += 1
            else:             burn_acc  += 1

        if it < burn_in and (it + 1) % 300 == 0 and it > 0:
            rate = burn_acc / (it + 1)
            if   rate < 0.15: prop_std *= 0.70
            elif rate > 0.50: prop_std *= 1.40

        # STEP 4 — θᵢ (Pólya-urn DP)
        out_idx = np.where(z_cur == 0)[0]
        if len(out_idx) > 0:
            for pos in np.random.permutation(len(out_idx)):
                i   = out_idx[pos]
                y_i = y_int[i]
                oth = out_idx[out_idx != i]

                if len(oth) == 0:
                    theta_cur[i] = np.random.gamma(a0 + y_i, 1.0 / (b0 + 1.0))
                else:
                    th_oth = theta_cur[oth]
                    uniq, cnts = np.unique(th_oth, return_counts=True)
                    lw_ex  = (np.log(cnts.astype(float))
                              + poisson.logpmf(y_i, np.maximum(uniq, 1e-10)))
                    lw_new = (np.log(alpha0_dp)
                              + nbinom.logpmf(y_i, a0, b0 / (1.0 + b0)))
                    lw     = np.append(lw_ex, lw_new)
                    lw    -= lw.max()
                    pr     = np.exp(lw); pr /= pr.sum()
                    ch     = np.random.choice(len(pr), p=pr)
                    theta_cur[i] = (uniq[ch] if ch < len(uniq)
                                    else np.random.gamma(a0 + y_i,
                                                         1.0 / (b0 + 1.0)))

            th_out = theta_cur[out_idx]
            uniq, inv = np.unique(th_out, return_inverse=True)
            for k in range(len(uniq)):
                mem = out_idx[inv == k]
                theta_cur[mem] = np.random.gamma(
                    a0 + y_fl[mem].sum(), 1.0 / (b0 + len(mem)))

        cln = np.where(z_cur == 1)[0]
        if len(cln) > 0:
            theta_cur[cln] = np.random.gamma(a0, 1.0 / b0, size=len(cln))

        if it >= burn_in:
            s_          = it - burn_in
            beta_samp[s_] = beta_cur
            z_samp[s_]    = z_cur

    acc_rate = acc_count / n_save if n_save > 0 else 0.0
    return dict(beta_samples=beta_samp, z_samples=z_samp,
                acceptance_rate=acc_rate)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SIMULATION STUDY
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation():
    rng_master = np.random.default_rng(MASTER_SEED)
    all_rows    = []
    rep_datasets = {}

    for c_pct in CONTAM_LEVELS:
        print(f"\n── Contamination = {c_pct}% ──────────────────────────────────")
        t_level = time.time()

        est_mle   = []
        est_dpd   = {a: [] for a in DPD_ALPHAS}
        est_bayes = []

        for rep in range(N_REP):
            seed_rep = int(rng_master.integers(0, 2**31))
            np.random.seed(seed_rep)
            rng_rep = np.random.default_rng(seed_rep)

            X, y, is_out = generate_data(N, BETA_TRUE, c_pct, OUTLIER_MEAN, rng_rep)

            try:    bm = fit_mle(X, y)
            except: bm = np.full(2, np.nan)
            est_mle.append(bm)

            for a in DPD_ALPHAS:
                try:    bd = fit_dpd(X, y, alpha=a, beta_init=bm)
                except: bd = np.full(2, np.nan)
                est_dpd[a].append(bd)

            try:
                out = run_mcmc(X, y,
                               n_iter=N_ITER, burn_in=BURN_IN,
                               alpha_p=ALPHA_P, beta_p=BETA_P, prior_var=PRIOR_VAR,
                               a0=A0, b0=B0, alpha0_dp=ALPHA0_DP,
                               proposal_std=0.20)
                pm = out['beta_samples'].mean(0)
                p_out_i = 1.0 - out['z_samples'].mean(0)
            except Exception:
                pm      = np.full(2, np.nan)
                p_out_i = np.full(N, np.nan)

            est_bayes.append(pm)

            if rep == 0:
                rep_datasets[c_pct] = dict(
                    X=X, y=y, is_out=is_out,
                    bm=bm,
                    dpd={a: fit_dpd(X, y, a, bm) for a in DPD_ALPHAS},
                    bayes_pm=pm,
                    p_out=p_out_i
                )

            if (rep + 1) % 10 == 0:
                elapsed = time.time() - t_level
                eta = elapsed / (rep + 1) * (N_REP - rep - 1)
                print(f"  rep {rep+1:3d}/{N_REP}  "
                      f"elapsed {elapsed/60:.1f}min  ETA {eta/60:.1f}min")

        def brmse(lst):
            a = np.array(lst)
            bias = np.nanmean(a - BETA_TRUE, 0)
            rmse = np.sqrt(np.nanmean((a - BETA_TRUE)**2, 0))
            return bias, rmse

        bias_mle, rmse_mle = brmse(est_mle)
        dpd_m = {a: brmse(est_dpd[a]) for a in DPD_ALPHAS}
        bias_b, rmse_b = brmse(est_bayes)

        fmt = "  {:<22} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}"
        print(f"\n  {'Method':<22} {'Bias(β₀)':>10} {'Bias(β₁)':>10} "
              f"{'RMSE(β₀)':>10} {'RMSE(β₁)':>10}")
        print("  " + "─"*62)
        print(fmt.format("MLE", bias_mle[0], bias_mle[1], rmse_mle[0], rmse_mle[1]))
        for a in DPD_ALPHAS:
            bv, rv = dpd_m[a]
            print(fmt.format(f"DPD(α={a})", bv[0], bv[1], rv[0], rv[1]))
        print(fmt.format("Bayes (proposed)", bias_b[0], bias_b[1], rmse_b[0], rmse_b[1]))

        for method, bv, rv in (
            [("MLE", bias_mle, rmse_mle)]
            + [(f"DPD_a{int(a*100):02d}", *dpd_m[a]) for a in DPD_ALPHAS]
            + [("Bayes", bias_b, rmse_b)]
        ):
            all_rows.append(dict(
                contamination=c_pct, method=method,
                bias_b0=round(float(bv[0]),5), bias_b1=round(float(bv[1]),5),
                rmse_b0=round(float(rv[0]),5), rmse_b1=round(float(rv[1]),5),
            ))

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUTDIR, "simulation_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSimulation table saved: {csv_path}")
    return df, rep_datasets


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SIMULATION PLOTS
# ══════════════════════════════════════════════════════════════════════════════

C = dict(clean='#2980B9', outlier='#E74C3C', flagged='#E67E22',
         true_line='black', mle_line='#27AE60', bayes_line='#E74C3C')
FIGSIZE_WIDE = (10, 5)
DPI_SAVE = 200

def fig1_scatter_fitted(rep_dataset, suffix=""):
    X, y = rep_dataset['X'], rep_dataset['y']
    is_out = rep_dataset['is_out']
    bm = rep_dataset['bm']
    dpd_25 = rep_dataset['dpd'][0.25]
    bayes_pm = rep_dataset['bayes_pm']
    x_vals = X[:, 1]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.scatter(x_vals[~is_out], y[~is_out], c=C['clean'], s=40, alpha=0.65, label='Clean')
    ax.scatter(x_vals[is_out], y[is_out], c=C['outlier'], s=100, marker='*', alpha=0.9,
               label='True outliers')
    xg = np.linspace(-2.15, 2.15, 400)
    Xg = np.column_stack([np.ones(400), xg])
    ax.plot(xg, np.exp(Xg @ BETA_TRUE), color=C['true_line'], lw=2.5,
            label=f'True: β=[{BETA_TRUE[0]}, {BETA_TRUE[1]}]')
    ax.plot(xg, np.exp(Xg @ bm), color=C['mle_line'], lw=1.8, ls='--',
            label=f'MLE: β=[{bm[0]:.2f}, {bm[1]:.2f}]')
    ax.plot(xg, np.exp(Xg @ dpd_25), color='#9B59B6', lw=1.8, ls='-.',
            label=f'DPD(α=0.25): β=[{dpd_25[0]:.2f}, {dpd_25[1]:.2f}]')
    ax.plot(xg, np.exp(Xg @ bayes_pm), color=C['bayes_line'], lw=2.2,
            label=f'Bayes (PM): β=[{bayes_pm[0]:.2f}, {bayes_pm[1]:.2f}]')
    ax.set_ylim(-0.5, min(float(y.max()) + 3, 40))
    ax.set_xlabel('Covariate $x$', fontsize=12)
    ax.set_ylabel('Count $Y$', fontsize=12)
    ax.set_title(f'Contaminated Poisson Regression — Fitted Curves\n'
                 f'(n={N}, {suffix.replace("_eps","") if suffix else "10"}% contamination)')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        path = os.path.join(OUTDIR, f'fig1_scatter_fitted{suffix}.{ext}')
        fig.savefig(path, dpi=DPI_SAVE, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: fig1_scatter_fitted{suffix}  (png + pdf)")


def fig2_prob_not_outlier(rep_dataset, suffix=""):
    X, y = rep_dataset['X'], rep_dataset['y']
    is_out = rep_dataset['is_out']
    p_out = rep_dataset['p_out']
    p_good = 1.0 - p_out
    x_vals = X[:, 1]
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.scatter(x_vals[~is_out], p_good[~is_out], c=C['clean'], s=40, alpha=0.65, label='Clean')
    ax.scatter(x_vals[is_out], p_good[is_out], c=C['outlier'], s=100, marker='*', alpha=0.9,
               label='True outliers')
    ax.axhline(0.5, color='k', lw=1.5, ls='--', label='Decision boundary (0.50)', alpha=0.7)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Covariate $x$', fontsize=12)
    ax.set_ylabel(r'$P(Z_i = 1 \mid \mathbf{y})$', fontsize=12)
    ax.set_title('Posterior Probability of Not Being an Outlier vs Covariate')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        path = os.path.join(OUTDIR, f'fig2_prob_not_outlier{suffix}.{ext}')
        fig.savefig(path, dpi=DPI_SAVE, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: fig2_prob_not_outlier{suffix}  (png + pdf)")


def fig3_true_vs_flagged(rep_dataset, suffix=""):
    X, y = rep_dataset['X'], rep_dataset['y']
    is_out = rep_dataset['is_out']
    p_out = rep_dataset['p_out']
    x_vals = X[:, 1]
    flagged = p_out >= 0.5
    not_flagged = ~flagged
    TN = ~is_out &  not_flagged
    TP =  is_out &  flagged
    FP = ~is_out &  flagged
    FN =  is_out &  not_flagged
    sens = TP.sum() / is_out.sum() if is_out.sum() > 0 else float('nan')
    spec = TN.sum() / (~is_out).sum() if (~is_out).sum() > 0 else float('nan')
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.scatter(x_vals[TN], y[TN], c='#2980B9', s=40, alpha=0.60, marker='o',
               label=f'True clean, not flagged (n={TN.sum()})')
    ax.scatter(x_vals[TP], y[TP], c='#E74C3C', s=120, alpha=0.90, marker='*',
               label=f'True outlier, flagged ✓ (n={TP.sum()})')
    ax.scatter(x_vals[FP], y[FP], c='#E67E22', s=80, alpha=0.85, marker='^',
               label=f'True clean, flagged (n={FP.sum()})')
    ax.scatter(x_vals[FN], y[FN], c='#8E44AD', s=100, alpha=0.85, marker='X',
               label=f'True outlier, missed (n={FN.sum()})')
    ax.set_xlabel('Covariate $x$', fontsize=12)
    ax.set_ylabel('Count $Y$', fontsize=12)
    ax.set_title(f'True Outliers vs Model-Flagged\n'
                 f'Sens={sens:.2f}, Spec={spec:.2f}  (P(Z=0|y)≥0.5)')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        path = os.path.join(OUTDIR, f'fig3_true_vs_flagged{suffix}.{ext}')
        fig.savefig(path, dpi=DPI_SAVE, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: fig3_true_vs_flagged{suffix}  (png + pdf)")


def fig4_bias_rmse(df_sim):
    cont = sorted(df_sim['contamination'].unique())
    methods = {
        'MLE'      : ('#27AE60', 's', 2.0, 8),
        'DPD_a10'  : ('#E67E22', 'o', 1.6, 6),
        'DPD_a25'  : ('#9B59B6', 'o', 1.6, 6),
        'DPD_a50'  : ('#1ABC9C', 'o', 1.6, 6),
        'Bayes'    : ('#E74C3C', '^', 2.5, 9),
    }
    labels = {
        'MLE'    : 'MLE',
        'DPD_a10': 'DPD (α=0.10)',
        'DPD_a25': 'DPD (α=0.25)',
        'DPD_a50': 'DPD (α=0.50)',
        'Bayes'  : 'Bayes (proposed)',
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for pi in range(2):
        pl = 'β₀' if pi == 0 else 'β₁'
        for mi, (metric, yl) in enumerate([('bias', 'Bias'), ('rmse', 'RMSE')]):
            ax = axes[mi][pi]
            col_name = f'{metric}_b{pi}'
            for mkey, (col, mk, lw, ms) in methods.items():
                vals = []
                for c in cont:
                    row = df_sim.loc[(df_sim['contamination'] == c) & (df_sim['method'] == mkey)]
                    vals.append(float(row[col_name].iloc[0]) if len(row) > 0 else np.nan)
                ax.plot(cont, vals, marker=mk, lw=lw, ms=ms, color=col, label=labels[mkey])
            ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)
            ax.set_xlabel('Contamination (%)', fontsize=10)
            ax.set_ylabel(f'{yl} ({pl})', fontsize=10)
            ax.set_xticks(cont)
            letters = [['(a)', '(b)'], ['(c)', '(d)']]
            ax.set_title(f'{letters[mi][pi]} {yl} — {pl}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    plt.suptitle(f'Simulation Study: Bias and RMSE\n'
                 f'$n$={N}, {N_REP} replications, outliers $\\sim$ Poisson({int(OUTLIER_MEAN)}), '
                 f'true $\\beta$ = {list(BETA_TRUE)}', fontsize=11)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        path = os.path.join(OUTDIR, f'fig4_bias_rmse.{ext}')
        fig.savefig(path, dpi=DPI_SAVE, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: fig4_bias_rmse  (png + pdf)")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  REAL DATA ANALYSIS  (custom CSV + epilepsy) with numerical summaries
# ══════════════════════════════════════════════════════════════════════════════

def load_real_data(csv_path, x_col='x', y_col='y'):
    """Load a two‑column CSV (x, y) and form design matrix (intercept + x)."""
    df = pd.read_csv(csv_path)
    x = df[x_col].values
    y = df[y_col].values.astype(float)
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    return X, y, x   # return x for plotting

def analyse_real_data(X, y, x_col=1):
    """Fit MLE, DPD, Bayes; returns dict with posterior summaries."""
    bm = fit_mle(X, y)
    dpd_fits = {a: fit_dpd(X, y, a) for a in DPD_ALPHAS}
    out = run_mcmc(X, y,
                   n_iter=N_ITER, burn_in=BURN_IN,
                   alpha_p=ALPHA_P, beta_p=BETA_P, prior_var=PRIOR_VAR,
                   a0=A0, b0=B0, alpha0_dp=ALPHA0_DP,
                   proposal_std=0.20)
    pm = out['beta_samples'].mean(0)
    p_out = 1.0 - out['z_samples'].mean(0)
    # Bayesian estimate of p (clean probability)
    p_clean = out['z_samples'].mean()
    return dict(X=X, y=y, bm=bm, dpd=dpd_fits, bayes_pm=pm, p_out=p_out,
                x_vals=X[:, x_col], plot_x_idx=x_col, p_clean=p_clean,
                cov_names=None)   # cov_names will be set later if needed

def _make_grid_for_curve(X, plot_x_idx, n_grid=400):
    p = X.shape[1]
    Xg = np.tile(X.mean(axis=0), (n_grid, 1))
    x_min, x_max = X[:, plot_x_idx].min(), X[:, plot_x_idx].max()
    grid = np.linspace(x_min, x_max, n_grid)
    Xg[:, plot_x_idx] = grid
    Xg[:, 0] = 1.0   # intercept
    return Xg, grid


# Generic real-data plotting functions
def fig_real_fitted(data, suffix="_real", covariate_label="x"):
    X = data['X']; y = data['y']
    plot_x_idx = data['plot_x_idx']
    bm = data['bm']; dpd_25 = data['dpd'][0.25]; bayes_pm = data['bayes_pm']
    x_vals = data['x_vals']
    Xg, grid = _make_grid_for_curve(X, plot_x_idx)
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.scatter(x_vals, y, c='gray', alpha=0.5, label='Observations')
    ax.plot(grid, np.exp(Xg @ bm), color='#27AE60', ls='--', lw=2,
            label=f'MLE β={tuple(round(v,2) for v in bm)}')
    ax.plot(grid, np.exp(Xg @ dpd_25), color='#9B59B6', ls='-.', lw=2,
            label=f'DPD(0.25) β={tuple(round(v,2) for v in dpd_25)}')
    ax.plot(grid, np.exp(Xg @ bayes_pm), color='#E74C3C', lw=2,
            label=f'Bayes β={tuple(round(v,2) for v in bayes_pm)}')
    ax.set_xlabel(covariate_label); ax.set_ylabel('y')
    ax.set_title('Fitted curves (other covariates held at mean)')
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    for ext in ['png', 'pdf']:
        path = os.path.join(OUTDIR, f'fig1{suffix}_fitted.{ext}')
        fig.savefig(path, dpi=DPI_SAVE, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: fig1{suffix}_fitted  (png + pdf)")

def fig_real_prob(data, suffix="_real", covariate_label="x"):
    p_good = 1.0 - data['p_out']; x_vals = data['x_vals']
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    sc = ax.scatter(x_vals, p_good, c=p_good, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label='P(not outlier)')
    ax.axhline(0.5, color='k', ls='--')
    ax.set_xlabel(covariate_label); ax.set_ylabel('P(Z=1|data)')
    ax.set_title('Posterior probability of being clean'); ax.grid(alpha=0.3)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        path = os.path.join(OUTDIR, f'fig2{suffix}_prob.{ext}')
        fig.savefig(path, dpi=DPI_SAVE, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: fig2{suffix}_prob  (png + pdf)")

def fig_real_flagged(data, suffix="_real", covariate_label="x"):
    flagged = data['p_out'] >= 0.5; x_vals = data['x_vals']; y = data['y']
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.scatter(x_vals[~flagged], y[~flagged], c='#2980B9', label='Model says clean', alpha=0.6)
    ax.scatter(x_vals[flagged], y[flagged], c='#E74C3C', marker='*', s=100,
               label=f'Flagged as outlier (n={flagged.sum()})')
    ax.set_xlabel(covariate_label); ax.set_ylabel('y')
    ax.set_title('Outlier detection (P(outlier)≥0.5)'); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        path = os.path.join(OUTDIR, f'fig3{suffix}_flagged.{ext}')
        fig.savefig(path, dpi=DPI_SAVE, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: fig3{suffix}_flagged  (png + pdf)")

def fig_real_diagnostic(data, suffix="_real", covariate_label="x"):
    p_out = data['p_out']; x_vals = data['x_vals']; y = data['y']
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.scatter(x_vals, y, s=50*p_out+20, c=p_out, cmap='hot', alpha=0.7,
               edgecolors='k', linewidth=0.3)
    ax.set_xlabel(covariate_label); ax.set_ylabel('y')
    ax.set_title('Observation weight: larger/darker = more likely outlier')
    cbar = plt.colorbar(ax.collections[0], ax=ax, label='P(outlier)')
    ax.grid(alpha=0.3); plt.tight_layout()
    for ext in ['png', 'pdf']:
        path = os.path.join(OUTDIR, f'fig4{suffix}_diagnostic.{ext}')
        fig.savefig(path, dpi=DPI_SAVE, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: fig4{suffix}_diagnostic  (png + pdf)")


# ── Numerical summary and CSV export for real data ──────────────────────────

def summarise_real_results(data, covariate_names=None, dataset_label="real_data"):
    """
    Create a DataFrame of coefficient estimates and Bayesian outlier summary,
    then print and save to CSV.
    """
    methods = []
    # MLE
    methods.append(('MLE', data['bm'], None, None))
    # DPD for each alpha
    for a in DPD_ALPHAS:
        methods.append((f'DPD(α={a})', data['dpd'][a], None, None))
    # Bayes
    methods.append(('Bayes', data['bayes_pm'], data['p_clean'], (data['p_out'] >= 0.5).sum()))

    # Determine column names
    p = len(data['bm'])
    if covariate_names is not None and len(covariate_names) == p:
        col_names = covariate_names
    else:
        col_names = [f'β{i}' for i in range(p)]

    rows = []
    for name, beta, p_clean, n_flagged in methods:
        row = {'Method': name}
        for i, b in enumerate(beta):
            row[col_names[i]] = round(b, 4)
        if p_clean is not None:
            row['P(clean) posterior mean'] = round(p_clean, 4)
            row['N flagged (P(outlier)≥0.5)'] = n_flagged
        rows.append(row)

    df = pd.DataFrame(rows)

    # Print to console
    print(f"\n{'='*60}")
    print(f"Numerical results for {dataset_label}")
    print(df.to_string(index=False))

    # Save CSV
    csv_path = os.path.join(OUTDIR, f'{dataset_label}_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    return df


# ── Thall & Vail (1990) epilepsy data — full longitudinal dataset ──────────

def load_epilepsy_data():
    """
    Returns X (intercept, Trt, Base, Age), y (236 seizure counts),
    and variable names. The data are the complete 4‑visit Thall & Vail (1990)
    epilepsy study. Each subject contributes four 2‑week counts.

    Note: This model treats visits as independent; no within‑subject
    correlation is accounted for.
    """
    # Each row: [Trt, Base, Age, y1, y2, y3, y4]
    raw = np.array([
        [1, 11, 31, 5, 3, 3, 3],
        [1, 11, 30, 3, 5, 3, 3],
        [1,  6, 25, 2, 4, 0, 5],
        [1,  8, 36, 4, 4, 1, 4],
        [1, 66, 22, 7, 18, 9, 21],
        [1, 27, 29, 5, 2, 8, 7],
        [1, 12, 31, 6, 4, 0, 2],
        [1, 52, 42,40, 20, 23, 12],
        [1, 23, 37, 5, 6, 6, 5],
        [1, 10, 28, 2, 2, 2, 2],
        [1, 52, 36,12, 5, 7, 8],
        [1, 18, 24, 3, 5, 5, 6],
        [1, 42, 41, 2, 3, 7, 4],
        [1, 87, 42,25, 15, 10, 14],
        [1, 50, 26,18, 13, 10, 9],
        [1, 18, 27, 4, 2, 4, 4],
        [1,111, 25,22, 18, 24, 21],
        [1, 18, 25, 3, 2, 3, 3],
        [1, 20, 22, 2, 3, 2, 4],
        [1, 12, 35, 3, 4, 3, 4],
        [1, 16, 37, 7, 7, 2, 5],
        [1, 22, 26,21, 13, 12, 8],
        [1, 24, 36,33, 21, 21, 12],
        [1, 14, 38,16, 11, 12, 6],
        [1, 18, 31, 2, 0, 0, 1],
        [1, 16, 33, 4, 2, 0, 0],
        [1, 22, 28, 8, 1, 5, 7],
        [1, 30, 28, 1, 2, 0, 0],
        [1, 80, 30,10, 8, 3, 5],
        [1, 42, 24, 9, 10, 2, 7],
        [1, 38, 28, 1, 2, 1, 2],
        [0, 11, 31, 3, 4, 1, 5],
        [0, 15, 28, 3, 6, 1, 4],
        [0, 27, 28, 3, 3, 3, 2],
        [0,  5, 24, 3, 2, 1, 4],
        [0, 12, 27, 5, 6, 1, 3],
        [0, 20, 22, 4, 4, 1, 4],
        [0, 16, 27,21, 14, 8, 8],
        [0, 31, 24, 3, 2, 0, 0],
        [0,  8, 42, 1, 0, 0, 3],
        [0, 26, 20, 2, 1, 1, 2],
        [0, 69, 20, 1, 4, 3, 0],
        [0, 19, 24, 3, 1, 3, 4],
        [0, 18, 18, 3, 2, 2, 3],
        [0, 16, 38, 2, 2, 0, 3],
        [0, 32, 32, 5, 1, 1, 3],
        [0, 18, 32, 2, 0, 1, 0],
        [0, 38, 23, 1, 3, 0, 2],
        [0, 50, 34,11, 5, 5, 6],
        [0, 33, 24, 0, 2, 0, 0],
        [0, 12, 22, 5, 4, 0, 3],
        [0, 23, 27, 4, 1, 4, 0],
        [0,  9, 32, 3, 1, 0, 1],
        [0, 35, 30, 2, 1, 1, 3],
        [0, 31, 24, 2, 0, 1, 1],
        [0, 67, 23, 2, 4, 5, 1],
        [0, 12, 20, 5, 4, 2, 3],
        [0, 46, 32, 5, 6, 0, 1],
        [0, 44, 32, 1, 6, 2, 2]
    ])

    # Expand to long format (59 subjects × 4 visits)
    Trt  = raw[:, 0]       # 1 = progabide, 0 = placebo
    Base = raw[:, 1]       # baseline 2-week seizure count
    Age  = raw[:, 2]
    y_long = raw[:, 3:7].ravel()   # 236 seizure counts

    n_subj = len(raw)
    Trt_long  = np.repeat(Trt, 4)
    Base_long = np.repeat(Base, 4)
    Age_long  = np.repeat(Age, 4)

    X = np.column_stack([np.ones(len(y_long)), Trt_long, Base_long, Age_long])
    var_names = ['Intercept', 'Treatment (1=progabide)', 'Baseline count',
                 'Age (years)']
    return X, y_long.astype(float), var_names


# ══════════════════════════════════════════════════════════════════════════════
# 7.  SUMMARY TABLE  (simulation)
# ══════════════════════════════════════════════════════════════════════════════

def print_and_save_table(df_sim):
    label_map = {
        'MLE'    : 'MLE',
        'DPD_a10': 'DPD (α = 0.10)',
        'DPD_a25': 'DPD (α = 0.25)',
        'DPD_a50': 'DPD (α = 0.50)',
        'Bayes'  : 'Bayesian (proposed)',
    }
    print("\n" + "=" * 75)
    print("SIMULATION RESULTS — Bias and RMSE")
    print(f"n = {N},  {N_REP} replications,  "
          f"outliers ~ Poisson({int(OUTLIER_MEAN)}),  "
          f"true β = {list(BETA_TRUE)}")
    print("=" * 75)
    for c in sorted(df_sim['contamination'].unique()):
        sub = df_sim[df_sim['contamination'] == c]
        print(f"\n  Contamination = {c}%")
        print(f"  {'Method':<24} {'Bias(β₀)':>10} {'Bias(β₁)':>10} "
              f"{'RMSE(β₀)':>10} {'RMSE(β₁)':>10}")
        print("  " + "─" * 64)
        for mkey, lbl in label_map.items():
            row = sub[sub['method'] == mkey]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            print(f"  {lbl:<24} {r.bias_b0:>10.4f} {r.bias_b1:>10.4f} "
                  f"{r.rmse_b0:>10.4f} {r.rmse_b1:>10.4f}")
    print("\n" + "=" * 75)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t_total = time.time()

    # ── Simulation ────────────────────────────────────────────────────────
    df_sim, rep_datasets = run_simulation()
    print_and_save_table(df_sim)

    print("\nGenerating simulation figures ...")
    for c_pct, rdata in rep_datasets.items():
        suffix = f"_eps{c_pct}"
        fig1_scatter_fitted(rdata, suffix)
        fig2_prob_not_outlier(rdata, suffix)
        fig3_true_vs_flagged(rdata, suffix)
    fig4_bias_rmse(df_sim)

    # ── Custom real data (if file exists) ─────────────────────────────────
    if os.path.exists(REAL_DATA_PATH):
        print(f"\nCustom real data file '{REAL_DATA_PATH}' found. Analysing ...")
        X_real, y_real, _ = load_real_data(REAL_DATA_PATH)
        real_res = analyse_real_data(X_real, y_real, x_col=1)
        # Plots
        fig_real_fitted(real_res, suffix="_real", covariate_label="x")
        fig_real_prob(real_res, suffix="_real", covariate_label="x")
        fig_real_flagged(real_res, suffix="_real", covariate_label="x")
        fig_real_diagnostic(real_res, suffix="_real", covariate_label="x")
        # Numerical summary
        summarise_real_results(real_res, covariate_names=['Intercept', 'x'],
                               dataset_label="real_data")
    else:
        print(f"\nCustom real data file '{REAL_DATA_PATH}' not found. Skipping.")

    # ── Thall & Vail epilepsy data ───────────────────────────────────────
    print("\nRunning Thall & Vail (1990) epilepsy data (all 4 visits per subject) ...")
    X_ep, y_ep, ep_vars = load_epilepsy_data()
    ep_res = analyse_real_data(X_ep, y_ep, x_col=2)  # Baseline as plotting covariate
    # Plots
    fig_real_fitted(ep_res, suffix="_epilepsy",
                    covariate_label="Baseline seizure count (per 2 weeks)")
    fig_real_prob(ep_res, suffix="_epilepsy",
                  covariate_label="Baseline seizure count")
    fig_real_flagged(ep_res, suffix="_epilepsy",
                     covariate_label="Baseline seizure count")
    fig_real_diagnostic(ep_res, suffix="_epilepsy",
                        covariate_label="Baseline seizure count")
    # Numerical summary
    summarise_real_results(ep_res, covariate_names=ep_vars,
                           dataset_label="epilepsy")

    # ── Save pickle ───────────────────────────────────────────────────────
    pkl_path = os.path.join(OUTDIR, 'simulation_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(dict(df_sim=df_sim, rep_datasets=rep_datasets), f)
    print(f"\nPickle saved: {pkl_path}")

    elapsed = (time.time() - t_total) / 3600
    print(f"\n{'=' * 65}")
    print(f"ALL DONE — Total elapsed: {elapsed:.2f} hours")
    print(f"Results in: {OUTDIR}")
    print('=' * 65)