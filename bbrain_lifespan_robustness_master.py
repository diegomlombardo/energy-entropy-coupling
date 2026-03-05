# ============================================================
# brain_lifespan_ecm_world_pipeline.py
# ============================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.utils import resample
from scipy.stats import rankdata, norm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set(style="whitegrid", context="talk", palette="colorblind")

# ============================================================
# 1. CONNECTIVITY MATRIX GENERATOR
# ============================================================
def generate_connectivity(N=40, seed=None):
    if seed is not None:
        np.random.seed(seed)
    C = np.random.rand(N, N)
    C = (C + C.T)/2
    np.fill_diagonal(C, 0)
    C = C / np.max(np.abs(np.linalg.eigvals(C)))
    return C

# ============================================================
# 2. ECM SIMULATION (ENERGY-CONSTRAINED, STRESSED)
# ============================================================
def simulate_ECM(C, age, T=350, dt=0.02, alpha0=1.2, metabolic_decline=0.02,
                 beta=0.6, delta=0.05, kappa=2.5, noise=0.02):
    N = C.shape[0]
    omega = np.random.uniform(0.04, 0.07, N)
    z = np.random.randn(N) + 1j*np.random.randn(N)
    E = 1.0
    alpha = alpha0 - metabolic_decline*age
    row_sums = np.sum(C, axis=1)
    timepoints = int(T/dt)
    perturb_time = int(0.4*timepoints)
    threshold = 1.0
    recovered = False
    recovery_time = timepoints
    energy_series = []

    for t in range(timepoints):
        neural_energy = np.mean(np.abs(z)**2)
        dE = alpha - beta*neural_energy - delta*E
        E += dE*dt
        E = max(E, 0.01)

        coupling = 0.2*(C @ z - row_sums*z)
        dz = (0.02 + 1j*omega - np.abs(z)**2)*z
        dz += coupling - kappa*(1.0/E)*z
        dz += noise*(np.random.randn(N)+1j*np.random.randn(N))
        z += dz*dt

        if t == perturb_time:
            z *= 0.2

        if t > perturb_time and not recovered:
            if np.mean(np.abs(z)) > threshold:
                recovery_time = t - perturb_time
                recovered = True

        energy_series.append(E)

    cognition = 1.0 / (recovery_time + 1e-6)
    energy_var = np.var(energy_series)
    return cognition, energy_var

# ============================================================
# 3. FEP SIMULATION
# ============================================================
def simulate_FEP(C, age, T=350, dt=0.02, precision=1.0, noise=0.02):
    N = C.shape[0]
    omega = np.random.uniform(0.04, 0.07, N)
    z = np.random.randn(N) + 1j*np.random.randn(N)
    timepoints = int(T/dt)
    row_sums = np.sum(C, axis=1)
    perturb_time = int(0.4*timepoints)
    threshold = 1.0
    recovered = False
    recovery_time = timepoints
    error_series = []

    for t in range(timepoints):
        prediction = C @ z
        prediction_error = z - prediction
        error_series.append(np.mean(np.abs(prediction_error)))

        coupling = 0.2*(C @ z - row_sums*z)
        dz = (0.02 + 1j*omega - np.abs(z)**2)*z
        dz += coupling - precision*prediction_error
        dz += noise*(np.random.randn(N)+1j*np.random.randn(N))
        z += dz*dt

        if t == perturb_time:
            z *= 0.2

        if t > perturb_time and not recovered:
            if np.mean(np.abs(z)) > threshold:
                recovery_time = t - perturb_time
                recovered = True

    cognition = 1.0 / (recovery_time + 1e-6)
    error_var = np.var(error_series)
    return cognition, error_var

# ============================================================
# 4. DATASET GENERATION
# ============================================================
def generate_dataset(C, n_subjects=80):
    ages = np.linspace(10, 80, n_subjects)
    rows = []
    for age in ages:
        cog_ecm, energy_var = simulate_ECM(C, age)
        cog_fep, error_var = simulate_FEP(C, age)
        rows.append({
            "Age": age,
            "Cog_ECM": cog_ecm,
            "Cog_FEP": cog_fep,
            "Energy_Var": energy_var,
            "Error_Var": error_var
        })
    df = pd.DataFrame(rows)
    df["Age_c"] = df["Age"] - df["Age"].mean()
    df["Age2"] = df["Age_c"]**2
    return df

# ============================================================
# 5. CROSS-VALIDATION & PERMUTATION
# ============================================================
def cross_val_r2(X, y, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train, test in kf.split(X):
        model = LinearRegression().fit(X[train], y[train])
        pred = model.predict(X[test])
        scores.append(r2_score(y[test], pred))
    return np.mean(scores)

def permutation_test(X, y, n_perm=500):
    real_score = cross_val_r2(X, y)
    perm_scores = []
    for _ in range(n_perm):
        y_perm = np.random.permutation(y)
        perm_scores.append(cross_val_r2(X, y_perm))
    p_value = np.mean(np.array(perm_scores) >= real_score)
    return real_score, p_value

# ============================================================
# 6. MEDIATION BOOTSTRAP (ECM → Brain Age → Cognition)
# ============================================================
def mediation_bootstrap(df, mediator_col="Energy_Var", outcome_col="Cog_ECM", n_boot=1000):
    indirect_effects = []
    for _ in range(n_boot):
        sample = resample(df)
        a = LinearRegression().fit(sample[["Age"]], sample[[mediator_col]]).coef_[0][0]
        b = LinearRegression().fit(sample[["Age", mediator_col]], sample[[outcome_col]]).coef_[0][1]
        indirect_effects.append(a*b)
    indirect_effects = np.array(indirect_effects)
    return np.mean(indirect_effects), np.percentile(indirect_effects, 2.5), np.percentile(indirect_effects, 97.5)

# ============================================================
# 7. REGIME DIVERGENCE
# ============================================================
def regime_divergence(C):
    kappas = np.linspace(0.2, 3.5, 25)
    cognition_vals = []
    for k in kappas:
        cog, _ = simulate_ECM(C, age=60, kappa=k)
        cognition_vals.append(cog)
    curvature = np.gradient(np.gradient(cognition_vals))
    return np.max(np.abs(curvature))

# ============================================================
# 8. RUN SINGLE PIPELINE
# ============================================================
def run_pipeline(seed=0):
    np.random.seed(seed)
    C = generate_connectivity(N=40, seed=seed)
    df = generate_dataset(C)

    # ECM
    X_ecm = df[["Age", "Energy_Var"]].values
    y_ecm = df["Cog_ECM"].values
    r2_ecm, p_ecm = permutation_test(X_ecm, y_ecm)

    # FEP
    X_fep = df[["Age", "Error_Var"]].values
    y_fep = df["Cog_FEP"].values
    r2_fep, p_fep = permutation_test(X_fep, y_fep)

    # Mediation
    med_mean, med_low, med_high = mediation_bootstrap(df)

    # Regime divergence
    curvature_peak = regime_divergence(C)

    return {
        "ECM_R2": r2_ecm,
        "ECM_p": p_ecm,
        "FEP_R2": r2_fep,
        "FEP_p": p_fep,
        "ECM_med_mean": med_mean,
        "ECM_med_CI_low": med_low,
        "ECM_med_CI_high": med_high,
        "Regime_curvature_peak": curvature_peak
    }

# ============================================================
# 9. MULTI-SEED ROBUSTNESS + FIGURES
# ============================================================
def run_multiseed_pipeline(n_seeds=20):
    results = []
    for seed in range(n_seeds):
        print(f"Running seed {seed}")
        results.append(run_pipeline(seed))
    df_res = pd.DataFrame(results)

    # Figures
    fig, axes = plt.subplots(2, 2, figsize=(14,10))
    sns.histplot(df_res["ECM_R2"], kde=True, ax=axes[0,0], color="#1f77b4")
    axes[0,0].set_title("Panel A: ECM R² distribution")
    sns.histplot(df_res["FEP_R2"], kde=True, ax=axes[0,1], color="#ff7f0e")
    axes[0,1].set_title("Panel B: FEP R² distribution")
    sns.histplot(df_res["ECM_med_mean"], kde=True, ax=axes[1,0], color="#2ca02c")
    axes[1,0].set_title("Panel C: ECM → Cognition mediation")
    axes[1,0].axvline(df_res["ECM_med_mean"].mean(), color="black", linestyle="--")
    axes[1,1].plot(df_res["Regime_curvature_peak"], marker="o")
    axes[1,1].set_title("Panel D: Regime divergence (curvature peak)")

    plt.tight_layout()
    plt.savefig("FIGURE_ECM_FEP_PANELS.png", dpi=600)
    plt.show()

    print("\n=== ROBUSTNESS SUMMARY ===\n")
    print(df_res.describe())
    return df_res

# ============================================================
# 10. MAIN EXECUTION
# ============================================================
if __name__=="__main__":
    print("\n=== ENERGY-CONSTRAINED VS PREDICTIVE PIPELINE ===\n")
    df_results = run_multiseed_pipeline(n_seeds=20)
