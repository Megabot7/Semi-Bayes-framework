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

# MCMC settings (tune here for runtime adjustment)
N_ITER   = 12_000     # total MCMC iterations
BURN_IN  =  2_000     # burn-in (discarded)
# → 10 000 post-burn-in samples per replicate

# Prior hyperparameters
ALPHA_P   = 18        # Beta(18, 2) prior on p: mean = 0.90
BETA_P    =  2
PRIOR_VAR = 100.0     # σ²_β for N(0, σ²_β · I) prior on β
A0        =  1.0      # G0 = Gamma(a0, rate=b0): base measure for DP
B0        =  0.5      # G0 mean = a0/b0 = 2.0 (below outlier mean 25)
ALPHA0_DP =  1.0      # DP concentration parameter

print("=" * 65)
print("Bayesian Semiparametric Robust Poisson Regression")
print("Simulation Study following Jha (2025)")
print(f"n={N}  reps={N_REP}  MCMC={N_ITER}/{BURN_IN}  seed={MASTER_SEED}")
print(f"Output directory: {OUTDIR}")
print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_data(n, beta_true, contamination_pct, outlier_mean, rng):
    """
    Generate one contaminated Poisson regression dataset.

    Clean:   Y_i ~ Poisson(exp(β₀ + β₁ x_i)),   x_i ~ Uniform(−2, 2)
    Outlier: Y_i ~ Poisson(outlier_mean)

    Contamination scheme mirrors Jha (2025) §4: a fraction ε of observations
    are replaced by draws from a high-mean Poisson distribution.
    """
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
    """Vectorised (n × y_max+1) Poisson PMF matrix (log-space, then exponentiate)."""
    k       = np.arange(0, y_max + 1, dtype=float)
    log_fac = np.zeros(y_max + 1)
    log_fac[1:] = np.cumsum(np.log(np.arange(1, y_max + 1)))
    log_pmf = (k[None, :] * np.log(np.maximum(mu, 1e-300))[:, None]
               - mu[:, None] - log_fac[None, :])
    return np.exp(log_pmf)                      # shape (n, y_max+1)


def dpd_objective(beta, X, y, alpha):
    """
    DPD objective H_n(β) = (1/n) Σᵢ [A(μᵢ) − (1+1/α) f(yᵢ;μᵢ)^α]
    A(μ) = Σ_{k≥0} Poisson(k;μ)^{1+α}   (exact truncated sum)
    """
    n   = len(y)
    mu  = np.exp(np.clip(X @ beta, -20, 20))
    y_max = max(int(np.max(mu + 8*np.sqrt(np.maximum(mu, 1)) + 30)),
                int(y.max()) + 10)
    F   = _pmf_matrix(mu, y_max)
    A   = np.sum(F ** (1 + alpha), axis=1)
    fyi = F[np.arange(n), y.astype(int)]
    return np.mean(A - (1.0 + 1.0 / alpha) * fyi ** alpha)


def fit_mle(X, y):
    """MLE for Poisson regression (analytic gradient)."""
    def neg_ll_g(b):
        mu  = np.exp(np.clip(X @ b, -20, 20))
        return (-np.sum(y * np.log(np.maximum(mu, 1e-300)) - mu),
                -X.T @ (y - mu))
    res = minimize(neg_ll_g, np.zeros(X.shape[1]), jac=True, method='BFGS',
                   options={'maxiter': 5000, 'gtol': 1e-10})
    return res.x


def fit_dpd(X, y, alpha, beta_init=None):
    """Minimum DPD Estimator for Poisson regression."""
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
    """Log-posterior for β given Z=z (up to constant)."""
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

    # ── Initialise ──────────────────────────────────────────────────────────
    beta_cur  = fit_dpd(X, y, alpha=0.25)
    mu_init   = np.exp(np.clip(X @ beta_cur, -20, 20))
    pearson_r = (y - mu_init) / np.sqrt(np.maximum(mu_init, 1e-8))
    z_cur     = (pearson_r <= 2.576).astype(int)
    p_cur     = float(np.clip(z_cur.mean(), 0.50, 0.95))
    theta_cur = np.full(n, a0 / b0)
    idx_out0  = np.where(z_cur == 0)[0]
    if len(idx_out0) > 0:
        theta_cur[idx_out0] = np.maximum(y[idx_out0], a0 / b0)

    # ── Storage ─────────────────────────────────────────────────────────────
    beta_samp = np.zeros((n_save, pdim))
    z_samp    = np.zeros((n_save, n),   dtype=np.int8)
    prop_std  = proposal_std
    acc_count = 0      # post-burn-in
    burn_acc  = 0      # burn-in (for adaptation)

    # ── Main loop ───────────────────────────────────────────────────────────
    for it in range(n_iter):

        mu_cur = np.exp(np.clip(X @ beta_cur, -20, 20))

        # STEP 1 — Z_i (Gibbs) ──────────────────────────────────────────────
        lp_c   = (np.log(p_cur + 1e-300)
                  + poisson.logpmf(y_int, mu_cur))
        lp_o   = (np.log(1.0 - p_cur + 1e-300)
                  + poisson.logpmf(y_int, np.maximum(theta_cur, 1e-8)))
        p_cln  = np.clip(np.exp(lp_c - np.logaddexp(lp_c, lp_o)), 0.0, 1.0)
        z_cur  = (np.random.uniform(size=n) < p_cln).astype(int)

        # STEP 2 — p (Gibbs) ────────────────────────────────────────────────
        s     = int(z_cur.sum())
        p_cur = np.clip(np.random.beta(alpha_p + s, beta_p + n - s),
                        0.40, 0.9999)

        # STEP 3 — β (MH) ───────────────────────────────────────────────────
        b_prop = beta_cur + np.random.normal(0.0, prop_std, pdim)
        log_r  = (_log_post_beta(b_prop, X, y_fl, z_cur, prior_var)
                - _log_post_beta(beta_cur, X, y_fl, z_cur, prior_var))
        if np.log(np.random.uniform()) < log_r:
            beta_cur = b_prop
            if it >= burn_in: acc_count += 1
            else:             burn_acc  += 1

        # Adaptive tuning — separate burn-in counter avoids zero-reads bug
        if it < burn_in and (it + 1) % 300 == 0 and it > 0:
            rate = burn_acc / (it + 1)
            if   rate < 0.15: prop_std *= 0.70
            elif rate > 0.50: prop_std *= 1.40

        # STEP 4 — θᵢ (Pólya-urn DP, Neal 2000 Algorithm 2) ────────────────
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

            # Cluster-atom resampling (Neal 2000, step c)
            th_out = theta_cur[out_idx]
            uniq, inv = np.unique(th_out, return_inverse=True)
            for k in range(len(uniq)):
                mem = out_idx[inv == k]
                theta_cur[mem] = np.random.gamma(
                    a0 + y_fl[mem].sum(), 1.0 / (b0 + len(mem)))

        # Clean obs: sample from G0 to maintain chain structure
        cln = np.where(z_cur == 1)[0]
        if len(cln) > 0:
            theta_cur[cln] = np.random.gamma(a0, 1.0 / b0, size=len(cln))

        # Store post-burn-in
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
    rep_dataset = {}   # save one representative dataset for the three plots

    for c_pct in CONTAM_LEVELS:
        print(f"\n── Contamination = {c_pct}% ──────────────────────────────────")
        t_level = time.time()

        est_mle   = []
        est_dpd   = {a: [] for a in DPD_ALPHAS}
        est_bayes = []
        p_out_rep = None          # posterior P(outlier) for the rep dataset

        for rep in range(N_REP):
            seed_rep = int(rng_master.integers(0, 2**31))
            np.random.seed(seed_rep)
            rng_rep = np.random.default_rng(seed_rep)

            X, y, is_out = generate_data(N, BETA_TRUE, c_pct, OUTLIER_MEAN, rng_rep)

            # MLE
            try:    bm = fit_mle(X, y)
            except: bm = np.full(2, np.nan)
            est_mle.append(bm)

            # DPD
            for a in DPD_ALPHAS:
                try:    bd = fit_dpd(X, y, alpha=a, beta_init=bm)
                except: bd = np.full(2, np.nan)
                est_dpd[a].append(bd)

            # Bayesian MCMC
            try:
                out = run_mcmc(
                    X, y,
                    n_iter=N_ITER, burn_in=BURN_IN,
                    alpha_p=ALPHA_P, beta_p=BETA_P, prior_var=PRIOR_VAR,
                    a0=A0, b0=B0, alpha0_dp=ALPHA0_DP,
                    proposal_std=0.20
                )
                bs = out['beta_samples']
                pm = bs.mean(0)
                p_out_i = 1.0 - out['z_samples'].mean(0)
            except Exception as e:
                pm      = np.full(2, np.nan)
                p_out_i = np.full(N, np.nan)

            est_bayes.append(pm)

            # Save representative dataset (ε=10%, rep=0)
            if c_pct == 10 and rep == 0:
                rep_dataset = dict(
                    X=X, y=y, is_out=is_out,
                    bm=bm,
                    dpd={a: fit_dpd(X, y, a, bm) for a in DPD_ALPHAS},
                    bayes_pm=pm,
                    p_out=p_out_i
                )

            if (rep + 1) % 10 == 0:
                elapsed = time.time() - t_level
                eta     = elapsed / (rep + 1) * (N_REP - rep - 1)
                print(f"  rep {rep+1:3d}/{N_REP}  "
                      f"elapsed {elapsed/60:.1f}min  ETA {eta/60:.1f}min")

        # ── Aggregate metrics ────────────────────────────────────────────────
        def brmse(lst):
            a    = np.array(lst)
            bias = np.nanmean(a - BETA_TRUE, 0)
            rmse = np.sqrt(np.nanmean((a - BETA_TRUE)**2, 0))
            return bias, rmse

        bias_mle, rmse_mle = brmse(est_mle)
        dpd_m = {a: brmse(est_dpd[a]) for a in DPD_ALPHAS}
        bias_b, rmse_b = brmse(est_bayes)

        # Print summary
        fmt = "  {:<22} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}"
        print(f"\n  {'Method':<22} {'Bias(β₀)':>10} {'Bias(β₁)':>10} "
              f"{'RMSE(β₀)':>10} {'RMSE(β₁)':>10}")
        print("  " + "─"*62)
        print(fmt.format("MLE",
                         bias_mle[0], bias_mle[1], rmse_mle[0], rmse_mle[1]))
        for a in DPD_ALPHAS:
            bv, rv = dpd_m[a]
            print(fmt.format(f"DPD(α={a})", bv[0], bv[1], rv[0], rv[1]))
        print(fmt.format("Bayes (proposed)",
                         bias_b[0], bias_b[1], rmse_b[0], rmse_b[1]))

        # Store CSV rows
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

    # Save CSV
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUTDIR, "simulation_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSimulation table saved: {csv_path}")

    return df, rep_dataset


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PLOTS  (style follows Jha 2025)
# ══════════════════════════════════════════════════════════════════════════════

# ── Colour palette ─────────────────────────────────────────────────────────────
C = dict(
    clean   = '#2980B9',   # blue — clean observations
    outlier = '#E74C3C',   # red  — true outliers
    flagged = '#E67E22',   # orange — model-flagged
    true_line = 'black',
    mle_line  = '#27AE60',
    bayes_line= '#E74C3C',
)
FIGSIZE_WIDE = (10, 5)
FIGSIZE_SQ   = (6, 5)
DPI_SAVE     = 200


def fig1_scatter_fitted(rep_dataset):
    X, y       = rep_dataset['X'], rep_dataset['y']
    is_out     = rep_dataset['is_out']
    bm         = rep_dataset['bm']
    dpd_25     = rep_dataset['dpd'][0.25]
    bayes_pm   = rep_dataset['bayes_pm']
    x_vals     = X[:, 1]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    # Data points
    ax.scatter(x_vals[~is_out], y[~is_out],
               c=C['clean'], s=40, alpha=0.65, label='Clean observations', zorder=3)
    ax.scatter(x_vals[is_out], y[is_out],
               c=C['outlier'], s=100, marker='*', alpha=0.9,
               label='True outliers', zorder=4)

    # Fitted curves
    xg = np.linspace(-2.15, 2.15, 400)
    Xg = np.column_stack([np.ones(400), xg])

    ax.plot(xg, np.exp(Xg @ BETA_TRUE), color=C['true_line'], lw=2.5,
            label=f'True: β=[{BETA_TRUE[0]}, {BETA_TRUE[1]}]', zorder=5)
    ax.plot(xg, np.exp(Xg @ bm), color=C['mle_line'], lw=1.8,
            linestyle='--',
            label=f'MLE: β=[{bm[0]:.2f}, {bm[1]:.2f}]', zorder=5)
    ax.plot(xg, np.exp(Xg @ dpd_25), color='#9B59B6', lw=1.8,
            linestyle='-.',
            label=f'DPD(α=0.25): β=[{dpd_25[0]:.2f}, {dpd_25[1]:.2f}]', zorder=5)
    ax.plot(xg, np.exp(Xg @ bayes_pm), color=C['bayes_line'], lw=2.2,
            label=f'Bayes (PM): β=[{bayes_pm[0]:.2f}, {bayes_pm[1]:.2f}]', zorder=6)

    ax.set_ylim(-0.5, min(float(y.max()) + 3, 40))
    ax.set_xlabel('Covariate $x$', fontsize=12)
    ax.set_ylabel('Count $Y$', fontsize=12)
    ax.set_title(
        'Contaminated Poisson Regression — Fitted Curves\n'
        f'(Representative dataset: $n$={N}, 10% contamination, '
        f'outliers $\\sim$ Poisson({int(OUTLIER_MEAN)}))',
        fontsize=11)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        path = os.path.join(OUTDIR, f'fig1_scatter_fitted.{ext}')
        fig.savefig(path, dpi=DPI_SAVE, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: fig1_scatter_fitted  (png + pdf)")


def fig2_prob_not_outlier(rep_dataset):
    X, y     = rep_dataset['X'], rep_dataset['y']
    is_out   = rep_dataset['is_out']
    p_out    = rep_dataset['p_out']       # P(Z_i = 0 | data)
    p_good   = 1.0 - p_out               # P(Z_i = 1 | data) = P(not outlier)
    x_vals   = X[:, 1]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    ax.scatter(x_vals[~is_out], p_good[~is_out],
               c=C['clean'], s=40, alpha=0.65,
               label='Clean observations', zorder=3)
    ax.scatter(x_vals[is_out], p_good[is_out],
               c=C['outlier'], s=100, marker='*', alpha=0.9,
               label='True outliers', zorder=4)

    ax.axhline(0.5, color='k', lw=1.5, linestyle='--',
               label='Decision boundary (0.50)', alpha=0.7)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Covariate $x$', fontsize=12)
    ax.set_ylabel(r'$P(Z_i = 1 \mid \mathbf{y})$', fontsize=12)
    ax.set_title(
        'Posterior Probability of Not Being an Outlier vs Covariate\n'
        r'$P(Z_i=1\mid\mathbf{y})$ — higher values indicate clean observations',
        fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        path = os.path.join(OUTDIR, f'fig2_prob_not_outlier.{ext}')
        fig.savefig(path, dpi=DPI_SAVE, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: fig2_prob_not_outlier  (png + pdf)")


def fig3_true_vs_flagged(rep_dataset):
    X, y   = rep_dataset['X'], rep_dataset['y']
    is_out = rep_dataset['is_out']
    p_out  = rep_dataset['p_out']
    x_vals = X[:, 1]

    flagged   = p_out >= 0.5
    not_flagged = ~flagged

    # Four categories
    TN = ~is_out &  not_flagged   # True negative  (clean, model says clean)
    TP =  is_out &  flagged       # True positive  (outlier, model flags)
    FP = ~is_out &  flagged       # False positive (clean, model flags)
    FN =  is_out &  not_flagged   # False negative (outlier, model misses)

    sens = TP.sum() / is_out.sum() if is_out.sum() > 0 else float('nan')
    spec = TN.sum() / (~is_out).sum() if (~is_out).sum() > 0 else float('nan')

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    ax.scatter(x_vals[TN], y[TN], c='#2980B9',  s=40,  alpha=0.60,
               marker='o', label=f'True clean, not flagged (n={TN.sum()})')
    ax.scatter(x_vals[TP], y[TP], c='#E74C3C',  s=120, alpha=0.90,
               marker='*', label=f'True outlier, flagged ✓ (n={TP.sum()})')
    ax.scatter(x_vals[FP], y[FP], c='#E67E22',  s=80,  alpha=0.85,
               marker='^', label=f'True clean, flagged (false pos.) (n={FP.sum()})')
    ax.scatter(x_vals[FN], y[FN], c='#8E44AD',  s=100, alpha=0.85,
               marker='X', label=f'True outlier, missed (false neg.) (n={FN.sum()})')

    ax.set_xlabel('Covariate $x$', fontsize=12)
    ax.set_ylabel('Count $Y$', fontsize=12)
    ax.set_title(
        f'True Outliers vs Model-Flagged Observations\n'
        f'Sensitivity = {sens:.2f},  Specificity = {spec:.2f}  '
        f'(threshold $P(Z_i=0|\\mathbf{{y}}) \\geq 0.5$)',
        fontsize=11)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        path = os.path.join(OUTDIR, f'fig3_true_vs_flagged.{ext}')
        fig.savefig(path, dpi=DPI_SAVE, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: fig3_true_vs_flagged  (png + pdf)")


def fig4_bias_rmse(df_sim):
    cont    = sorted(df_sim['contamination'].unique())
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
                vals = [
                    float(df_sim.loc[
                        (df_sim['contamination'] == c)
                        & (df_sim['method'] == mkey), col_name
                    ].iloc[0]) if len(df_sim.loc[
                        (df_sim['contamination'] == c)
                        & (df_sim['method'] == mkey)]) > 0 else np.nan
                    for c in cont
                ]
                ax.plot(cont, vals, marker=mk, lw=lw, ms=ms,
                        color=col, label=labels[mkey])

            ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)
            ax.set_xlabel('Contamination (%)', fontsize=10)
            ax.set_ylabel(f'{yl} ({pl})', fontsize=10)
            ax.set_xticks(cont)
            letters = [['(a)', '(b)'], ['(c)', '(d)']]
            ax.set_title(f'{letters[mi][pi]} {yl} — {pl}',
                         fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

    plt.suptitle(
        f'Simulation Study: Bias and RMSE\n'
        f'$n$={N}, {N_REP} replications, outliers $\\sim$ Poisson({int(OUTLIER_MEAN)}), '
        f'true $\\beta$ = {list(BETA_TRUE)}',
        fontsize=11)
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        path = os.path.join(OUTDIR, f'fig4_bias_rmse.{ext}')
        fig.savefig(path, dpi=DPI_SAVE, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: fig4_bias_rmse  (png + pdf)")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  SUMMARY TABLE  (clean, printable)
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
# 7.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t_total = time.time()

    # ── Run simulation ────────────────────────────────────────────────────────
    df_sim, rep_dataset = run_simulation()

    # ── Print summary table ───────────────────────────────────────────────────
    print_and_save_table(df_sim)

    # ── Generate plots ────────────────────────────────────────────────────────
    print("\nGenerating figures ...")
    if rep_dataset:
        fig1_scatter_fitted(rep_dataset)
        fig2_prob_not_outlier(rep_dataset)
        fig3_true_vs_flagged(rep_dataset)
    else:
        print("  WARNING: representative dataset not captured — skipping Figs 1–3")

    fig4_bias_rmse(df_sim)

    # ── Save pickle for reproducibility ──────────────────────────────────────
    pkl_path = os.path.join(OUTDIR, 'simulation_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(dict(df_sim=df_sim, rep_dataset=rep_dataset), f)
    print(f"Pickle saved: {pkl_path}")

    elapsed = (time.time() - t_total) / 3600
    print(f"\n{'=' * 65}")
    print(f"ALL DONE — Total elapsed: {elapsed:.2f} hours")
    print(f"Results in: {OUTDIR}")
    print('=' * 65)
