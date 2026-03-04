# ============================================================
# Q1 CROSS-WORLD EECI SIMULATION - PUBLICATION READY
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
import os

sns.set(style="whitegrid", context="talk")
np.random.seed(42)

FIG_FOLDER = "./figures_q1"
os.makedirs(FIG_FOLDER, exist_ok=True)

# ============================================================
# 1. CONNECTIVITY
# ============================================================

def create_connectivity(N=40):
    """Random symmetric connectivity matrix normalized for stability."""
    C = np.random.rand(N, N)
    C = (C + C.T) / 2
    np.fill_diagonal(C, 0)
    return C / np.max(np.abs(np.linalg.eigvals(C)))

# ============================================================
# 2. MECHANISTIC ENERGY & PREDICTIVE DYNAMICS
# ============================================================

def simulate_EECI(C, age):
    N = C.shape[0]
    z = np.random.randn(N)
    reservoir = max(0.1, 1.5 * np.exp(-0.02 * age))
    for _ in range(200):
        neural_energy = np.mean(z**2)
        delta = 0.05 * (0.5 - 0.6*neural_energy - 0.05*reservoir)
        z = np.clip(z + 0.1*(C@z - z) - (1/reservoir)*z + np.random.randn(N)*0.01, -1e2, 1e2)
        reservoir = np.clip(reservoir + delta, 0.01, 5)
    return z

def simulate_PPI(C, age):
    N = C.shape[0]
    z = np.random.randn(N)
    for _ in range(200):
        prediction = C @ z
        z = np.clip(z - 0.8*(z - prediction) + np.random.randn(N)*0.01, -1e2, 1e2)
    return z

# ============================================================
# 3. COGNITION GENERATION (NON-CIRCULAR)
# ============================================================

def generate_cognition(world, eeci, ppi, snr=3):
    """
    Cognition partially depends on EECI/PPI with reduced weight + independent noise
    to avoid circularity.
    """
    weights = {"energy": 0.3, "predictive": 0.3, "mixed": 0.15}  # reduced influence
    signal = {
        "energy": weights["energy"]*np.mean(eeci),
        "predictive": weights["predictive"]*np.mean(ppi),
        "mixed": weights["mixed"]*np.mean(eeci) + weights["mixed"]*np.mean(ppi)
    }.get(world, None)
    
    if signal is None:
        raise ValueError("Unknown world type")
    
    noise_var = np.var([signal])/snr + 1e-8
    independent_noise = np.random.randn() * np.sqrt(noise_var) + np.random.randn()*0.5
    return signal + independent_noise

# ============================================================
# 4. LONGITUDINAL DATASET
# ============================================================

def generate_longitudinal(world, n_subjects=80, n_timepoints=5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    C = create_connectivity()
    rows = []
    for subj in range(n_subjects):
        baseline_age = np.random.uniform(10, 60)
        for t in range(n_timepoints):
            age = baseline_age + 5*t
            eeci_vec = simulate_EECI(C, age)
            ppi_vec = simulate_PPI(C, age)
            cog = generate_cognition(world, eeci_vec, ppi_vec)
            rows.append([subj, t, age, np.mean(eeci_vec), np.mean(ppi_vec), cog])
    return pd.DataFrame(rows, columns=["ID", "Time", "Age", "EECI", "PPI", "Cog"])

# ============================================================
# 5. REGRESSION ANALYSIS WITH MODEL COMPARISON
# ============================================================

def cross_world_regression(df):
    X = df[["EECI","PPI","Age"]].copy()
    X["Age2"] = X["Age"]**2
    Y = df["Cog"].values
    valid = ~np.isnan(Y) & X.notna().all(axis=1)
    X_s = X.loc[valid].values
    Y_s = Y[valid]

    model_full = LinearRegression().fit(X_s, Y_s)
    r2_full = model_full.score(X_s, Y_s)
    
    model_red = LinearRegression().fit(X_s[:,1:], Y_s)
    r2_red = model_red.score(X_s[:,1:], Y_s)
    delta_r2 = r2_full - r2_red
    
    # AIC / BIC (assuming Gaussian residuals)
    n = len(Y_s)
    k_full = X_s.shape[1]
    resid = Y_s - model_full.predict(X_s)
    sse = np.sum(resid**2)
    aic = n * np.log(sse/n) + 2*k_full
    bic = n * np.log(sse/n) + k_full*np.log(n)
    
    # Cross-validated R²
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = np.mean(cross_val_score(model_full, X_s, Y_s, cv=cv, scoring="r2"))

    return dict(
        Beta_EECI=model_full.coef_[0],
        Beta_PPI=model_full.coef_[1],
        Beta_Age=model_full.coef_[2],
        Beta_Age2=model_full.coef_[3],
        Intercept=model_full.intercept_,
        R2=r2_full,
        Delta_R2=delta_r2,
        AIC=aic,
        BIC=bic,
        CV_R2=cv_r2
    )

# ============================================================
# 6. ROBUSTNESS ACROSS MULTIPLE SEEDS
# ============================================================

def run_robustness(seeds=50):
    results = []
    print(f"Running robustness analysis with {seeds} seeds...\n")
    for world in ["energy","predictive","mixed"]:
        for seed in range(seeds):
            df = generate_longitudinal(world, seed=seed)
            res = cross_world_regression(df)
            res["World"] = world
            res["Seed"] = seed
            results.append(res)
        print(f"  Completed world: {world}")
    return pd.DataFrame(results)

# ============================================================
# 7. DISCRIMINANT VALIDITY
# ============================================================

def discriminant_validity(df):
    corr_matrix = df[["EECI","PPI"]].corr()
    print("\nDiscriminant validity (correlation between EECI and PPI):")
    print(corr_matrix)
    return corr_matrix

# ============================================================
# 8. PLOTTING FUNCTIONS
# ============================================================

def plot_beta_distributions(df_results):
    plt.figure(figsize=(8,6))
    sns.boxplot(x="World", y="Beta_EECI", data=df_results, palette="Set2")
    plt.title("Distribution of EECI Beta Coefficients Across Worlds")
    plt.savefig(os.path.join(FIG_FOLDER,"beta_eeci.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8,6))
    sns.boxplot(x="World", y="Beta_Age2", data=df_results, palette="Set3")
    plt.title("Distribution of Quadratic Age Coefficients Across Worlds")
    plt.savefig(os.path.join(FIG_FOLDER,"beta_age2.png"), dpi=300)
    plt.close()

def plot_longitudinal_trajectories(world, n_subjects=20, n_timepoints=5):
    df = generate_longitudinal(world, n_subjects=n_subjects, n_timepoints=n_timepoints, seed=42)
    plt.figure(figsize=(8,6))
    sns.lineplot(x="Age", y="EECI", hue="ID", data=df, alpha=0.5, legend=False)
    sns.lineplot(x="Age", y="EECI", data=df.groupby("Age").mean().reset_index(), color="black", lw=3)
    plt.title(f"Longitudinal EECI trajectories ({world} world)")
    plt.savefig(os.path.join(FIG_FOLDER,f"trajectories_EECI_{world}.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8,6))
    sns.lineplot(x="Age", y="Cog", hue="ID", data=df, alpha=0.5, legend=False)
    sns.lineplot(x="Age", y="Cog", data=df.groupby("Age").mean().reset_index(), color="black", lw=3)
    plt.title(f"Longitudinal Cognition trajectories ({world} world)")
    plt.savefig(os.path.join(FIG_FOLDER,f"trajectories_Cog_{world}.png"), dpi=300)
    plt.close()

# ============================================================
# 9. MAIN
# ============================================================

if __name__ == "__main__":
    # Run multi-seed robustness
    df_results = run_robustness(seeds=20)  # adjust seeds for publication
    print("\nRobustness analysis completed.")

    # Summary table
    summary = df_results.groupby("World").agg(
        Beta_EECI_mean=("Beta_EECI","mean"),
        Beta_EECI_lower=("Beta_EECI", lambda x: np.percentile(x, 2.5)),
        Beta_EECI_upper=("Beta_EECI", lambda x: np.percentile(x, 97.5)),
        Beta_PPI_mean=("Beta_PPI","mean"),
        Beta_PPI_lower=("Beta_PPI", lambda x: np.percentile(x, 2.5)),
        Beta_PPI_upper=("Beta_PPI", lambda x: np.percentile(x, 97.5)),
        Beta_Age_mean=("Beta_Age","mean"),
        Beta_Age_lower=("Beta_Age", lambda x: np.percentile(x, 2.5)),
        Beta_Age_upper=("Beta_Age", lambda x: np.percentile(x, 97.5)),
        Beta_Age2_mean=("Beta_Age2","mean"),
        Beta_Age2_lower=("Beta_Age2", lambda x: np.percentile(x, 2.5)),
        Beta_Age2_upper=("Beta_Age2", lambda x: np.percentile(x, 97.5)),
        Delta_R2_mean=("Delta_R2","mean"),
        Delta_R2_lower=("Delta_R2", lambda x: np.percentile(x, 2.5)),
        Delta_R2_upper=("Delta_R2", lambda x: np.percentile(x, 97.5)),
        AIC_mean=("AIC","mean"),
        BIC_mean=("BIC","mean"),
        CV_R2_mean=("CV_R2","mean")
    )
    print("\nSummary Table (mean ± 95% CI):")
    print(summary)

    # Plot beta distributions
    plot_beta_distributions(df_results)

    # Plot longitudinal trajectories for all worlds
    for world in ["energy","predictive","mixed"]:
        plot_longitudinal_trajectories(world, n_subjects=20)
    
    # Discriminant validity
    df_example = generate_longitudinal("energy", n_subjects=50, n_timepoints=5, seed=42)
    discriminant_validity(df_example)

    print(f"\nAll figures saved in: {FIG_FOLDER}")
