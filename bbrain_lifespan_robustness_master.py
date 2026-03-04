import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from dataclasses import dataclass

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    n_subjects: int = 120
    timepoints: int = 20
    n_regions: int = 35
    dt: float = 0.05
    base_kappa: float = 0.5
    seeds: int = 50
    permutations: int = 500
    n_splits: int = 5
    noise_scale: float = 0.1  # reduced for stable signal
    kappa_sweep: tuple = (0.3, 0.5, 0.7)

CFG = Config()
WORLDS = ["energy", "predictive", "mixed", "null"]

np.seterr(all="raise")  # detect numerical errors

# ============================================================
# CONNECTOME GENERATION
# ============================================================

def spectral_normalize(W):
    eigvals = np.linalg.eigvals(W)
    rho = np.max(np.abs(eigvals))
    return W / (rho + 1e-8)

def generate_connectome(rng, n):
    W = rng.normal(0, 1, (n, n))
    W = (W + W.T) / 2  # symmetric
    return spectral_normalize(W)

# ============================================================
# SUBJECT SIMULATION
# ============================================================

def simulate_subject(world, rng, cfg, kappa_scale=1.0):
    W = generate_connectome(rng, cfg.n_regions)
    z = rng.normal(0, 0.4, cfg.n_regions)
    rows = []

    for t in range(cfg.timepoints):
        age = 20 + t * 3
        kappa = cfg.base_kappa * kappa_scale
        dz = W @ z - kappa * z
        z = np.tanh(z + cfg.dt * dz)

        # Features
        energy = np.mean(z ** 2)
        pred_error = np.var(W @ z - z)

        # WORLD-SPECIFIC COGNITION
        if world == "energy":
            cognition = np.exp(energy)
            pred_error = rng.normal(0, 1)
        elif world == "predictive":
            cognition = np.tanh(pred_error)
            energy = rng.normal(0, 1)
        elif world == "mixed":
            cognition = np.exp(energy) + np.tanh(pred_error)
        elif world == "null":
            cognition = rng.normal(0, 1)
            energy = rng.normal(0, 1)
            pred_error = rng.normal(0, 1)

        # Age effect (simulate inverted-U)
        age_effect = 0.05 * (age - 20) - 0.001 * (age - 20) ** 2
        cognition += age_effect

        # Noise
        cognition += rng.normal(0, cfg.noise_scale)
        z += rng.normal(0, 0.05, cfg.n_regions)

        rows.append([age, energy, pred_error, cognition])

    return rows

# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset(world, seed, cfg, kappa_scale=1.0):
    rng = np.random.default_rng(seed)
    data = []
    for s in range(cfg.n_subjects):
        subject_rows = simulate_subject(world, rng, cfg, kappa_scale)
        for row in subject_rows:
            data.append([s] + row)
    df = pd.DataFrame(data, columns=["subject", "age", "energy", "pred_error", "cognition"])
    return df

# ============================================================
# CROSS-SECTIONAL REGRESSION
# ============================================================

def cross_sectional_r2(df, cfg):
    X = df[["energy", "pred_error"]].values
    y = df["cognition"].values
    groups = df["subject"].values
    X = StandardScaler().fit_transform(X)

    gkf = GroupKFold(n_splits=cfg.n_splits)
    r2_scores, betas = [], []

    for tr, te in gkf.split(X, y, groups):
        model = Ridge(alpha=1.0)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        r2_scores.append(r2_score(y[te], pred))
        betas.append(model.coef_)

    return np.mean(r2_scores), np.mean(betas, axis=0)

# ============================================================
# LONGITUDINAL REGRESSION (AGE EFFECT)
# ============================================================

def longitudinal_regression(df):
    slopes, early_energy = [], []
    for s in df.subject.unique():
        sub = df[df.subject == s]
        model = LinearRegression().fit(sub[["age"]], sub["cognition"])
        slopes.append(model.coef_[0])
        early_energy.append(sub[sub.age < 40]["energy"].mean())

    dfv = pd.DataFrame({"early_energy": early_energy, "vulnerability": slopes})
    model = LinearRegression().fit(dfv[["early_energy"]], dfv["vulnerability"])
    return model.coef_[0], model.score(dfv[["early_energy"]], dfv["vulnerability"])

# ============================================================
# PERMUTATION TEST
# ============================================================

def permutation_test(df, true_r2, cfg, seed):
    rng = np.random.default_rng(seed)
    subjects = df.subject.unique()
    null_scores = []

    for _ in range(cfg.permutations):
        shuffled = rng.permutation(subjects)
        mapping = dict(zip(subjects, shuffled))
        df_perm = df.copy()
        for orig, new in mapping.items():
            df_perm.loc[df.subject == orig, "cognition"] = df[df.subject == new]["cognition"].values
        r2, _ = cross_sectional_r2(df_perm, cfg)
        null_scores.append(r2)

    null_scores = np.array(null_scores)
    pval = (np.sum(null_scores >= true_r2) + 1) / (len(null_scores) + 1)
    return pval

# ============================================================
# WORLD EVALUATION
# ============================================================

def evaluate_world(world, cfg):
    cross_r2_list, long_coef_list, perm_sig_list, beta_list = [], [], [], []

    for seed in range(cfg.seeds):
        df = generate_dataset(world, seed, cfg)
        r2_cross, beta = cross_sectional_r2(df, cfg)
        long_coef, long_r2 = longitudinal_regression(df)
        pval = permutation_test(df, r2_cross, cfg, seed + 999)

        cross_r2_list.append(r2_cross)
        long_coef_list.append(long_coef)
        perm_sig_list.append(pval < 0.05)
        beta_list.append(beta)

    beta_array = np.array(beta_list)
    return {
        "World": world,
        "Cross-sectional R²": np.mean(cross_r2_list),
        "Cross-sectional SE": np.std(cross_r2_list),
        "Longitudinal Slope": np.mean(long_coef_list),
        "Longitudinal SE": np.std(long_coef_list),
        "Permutation (%)": 100 * np.mean(perm_sig_list),
        "Beta Energy": np.mean(beta_array[:, 0]),
        "Beta Predictive": np.mean(beta_array[:, 1])
    }

# ============================================================
# CROSS-WORLD GENERALIZATION
# ============================================================

def cross_world_matrix(cfg):
    matrix = {}
    for w1 in WORLDS:
        for w2 in WORLDS:
            r2s = []
            for seed in range(cfg.seeds):
                df_train = generate_dataset(w1, seed, cfg)
                df_test = generate_dataset(w2, seed + 999, cfg)
                X_train = StandardScaler().fit_transform(df_train[["energy", "pred_error"]])
                y_train = df_train["cognition"].values
                X_test = StandardScaler().fit_transform(df_test[["energy", "pred_error"]])
                y_test = df_test["cognition"].values
                model = Ridge(alpha=1.0).fit(X_train, y_train)
                pred = model.predict(X_test)
                r2s.append(r2_score(y_test, pred))
            matrix[(w1, w2)] = np.mean(r2s)
    return matrix

# ============================================================
# PLOTTING FUNCTION
# ============================================================

def plot_results(summary_df, cw_matrix):
    sns.set(style="whitegrid", context="talk")
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # Cross-sectional R²
    ax = axs[0, 0]
    sns.barplot(x="World", y="Cross-sectional R²", data=summary_df, yerr=summary_df["Cross-sectional SE"], capsize=0.1, ax=ax)
    ax.set_title("Cross-sectional R²")

    # Longitudinal Slope
    ax = axs[0, 1]
    sns.barplot(x="World", y="Longitudinal Slope", data=summary_df, yerr=summary_df["Longitudinal SE"], capsize=0.1, ax=ax)
    ax.set_title("Longitudinal Effect of Early Energy")

    # Cross-world heatmap
    ax = axs[1, 0]
    cw_df = pd.DataFrame.from_dict(cw_matrix, orient='index', columns=['R²'])
    cw_df[['Train', 'Test']] = pd.DataFrame(cw_df.index.tolist(), index=cw_df.index)
    pivot = cw_df.pivot(index='Train', columns='Test', values='R²')
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Cross-world Generalization R²")

    # Beta coefficients
    ax = axs[1, 1]
    summary_df.plot.bar(x="World", y=["Beta Energy", "Beta Predictive"], ax=ax)
    ax.set_title("Average Regression Coefficients")
    plt.tight_layout()
    plt.show()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    all_results = [evaluate_world(w, CFG) for w in WORLDS]
    summary_df = pd.DataFrame(all_results)
    print("\n=== FINAL SUMMARY TABLE ===\n")
    print(summary_df)

    print("\n=== CROSS-WORLD GENERALIZATION R² ===\n")
    cw_matrix = cross_world_matrix(CFG)
    for k, v in cw_matrix.items():
        print(f"{k}: {v:.3f}")

    plot_results(summary_df, cw_matrix)
