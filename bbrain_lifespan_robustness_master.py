# ============================================================
# Q1 ENERGY-CONSTRAINED COGNITION MASTER PIPELINE (STRESSED)
# ============================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.utils import resample
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONNECTIVITY
# ============================================================

def generate_connectivity(N, seed):
    np.random.seed(seed)
    C = np.random.rand(N, N)
    C = (C + C.T) / 2
    np.fill_diagonal(C, 0)
    # Normalize by largest eigenvalue for stability
    C = C / np.max(np.abs(np.linalg.eigvals(C)))
    return C

# ============================================================
# 2. ECM SIMULATION (NUMERICALLY STABLE)
# ============================================================

def simulate_ECM(C, age,
                 T=350, dt=0.02,
                 alpha0=1.2,
                 metabolic_decline=0.02,
                 beta=0.6,
                 delta=0.05,
                 kappa=2.5,
                 noise=0.02):

    N = C.shape[0]
    omega = np.random.uniform(0.04, 0.07, N)
    z = np.random.randn(N) + 1j*np.random.randn(N)
    E = 1.0

    alpha = alpha0 - metabolic_decline * age
    timepoints = int(T/dt)
    row_sums = np.sum(C, axis=1)

    perturb_time = int(0.4*timepoints)
    threshold = 1.0

    recovered = False
    recovery_time = timepoints
    energy_series = []

    for t in range(timepoints):

        neural_energy = np.mean(np.abs(z)**2)
        if np.isnan(neural_energy) or neural_energy > 1e6:
            return 0.0, 0.0  # abort if runaway

        dE = alpha - beta*neural_energy - delta*E
        E += dE*dt
        E = np.clip(E, 0.01, 5.0)

        coupling = 0.2*(C @ z - row_sums*z)

        dz = (0.02 + 1j*omega - np.abs(z)**2)*z
        dz += coupling
        dz += -kappa*(1.0/E)*z
        dz += noise*(np.random.randn(N)+1j*np.random.randn(N))

        z += dz*dt

        # Cap neural amplitudes
        z = np.clip(z.real, -50, 50) + 1j*np.clip(z.imag, -50, 50)

        if t == perturb_time:
            z *= 0.2

        if t > perturb_time and not recovered:
            if np.mean(np.abs(z)) > threshold:
                recovery_time = t - perturb_time
                recovered = True

        energy_series.append(E)

    cognition = 1.0 / (recovery_time + 1e-6)
    energy_var = np.var(energy_series)

    # final NaN check
    if np.isnan(cognition) or np.isnan(energy_var):
        return 0.0, 0.0

    return cognition, energy_var

# ============================================================
# 3. FEP SIMULATION (NUMERICALLY STABLE)
# ============================================================

def simulate_FEP(C, age,
                 T=350, dt=0.02,
                 precision=1.0,
                 noise=0.02):

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
        pe = np.mean(np.abs(prediction_error))

        if np.isnan(pe) or pe > 1e6:
            return 0.0, 0.0  # abort if runaway

        error_series.append(pe)

        coupling = 0.2*(C @ z - row_sums*z)

        dz = (0.02 + 1j*omega - np.abs(z)**2)*z
        dz += coupling
        dz += -precision*prediction_error
        dz += noise*(np.random.randn(N)+1j*np.random.randn(N))

        z += dz*dt

        # cap neural amplitudes
        z = np.clip(z.real, -50, 50) + 1j*np.clip(z.imag, -50, 50)

        if t == perturb_time:
            z *= 0.2

        if t > perturb_time and not recovered:
            if np.mean(np.abs(z)) > threshold:
                recovery_time = t - perturb_time
                recovered = True

    cognition = 1.0 / (recovery_time + 1e-6)
    error_var = np.var(error_series)

    if np.isnan(cognition) or np.isnan(error_var):
        return 0.0, 0.0

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

    return pd.DataFrame(rows)

# ============================================================
# 5. CROSS-VALIDATION + PERMUTATION
# ============================================================

def cross_validate_model(X, y):
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = []
    for train, test in kf.split(X):
        model = LinearRegression().fit(X[train], y[train])
        pred = model.predict(X[test])
        scores.append(r2_score(y[test], pred))
    return np.mean(scores)

def permutation_test(X, y, n_perm=500):
    real_score = cross_validate_model(X, y)
    perm_scores = []
    for _ in range(n_perm):
        y_perm = np.random.permutation(y)
        perm_scores.append(cross_validate_model(X, y_perm))
    p_value = np.mean(np.array(perm_scores) >= real_score)
    return real_score, p_value

# ============================================================
# 6. REGIME DIVERGENCE
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
# 7. MEDIATION
# ============================================================

def mediation_bootstrap(df, mediator_col, cog_col, n_boot=1000):

    indirect_effects = []

    for _ in range(n_boot):
        sample = resample(df)

        a = LinearRegression().fit(
            sample[["Age"]],
            sample[[mediator_col]]
        ).coef_[0][0]

        b = LinearRegression().fit(
            sample[["Age", mediator_col]],
            sample[[cog_col]]
        ).coef_[0][1]

        indirect_effects.append(a*b)

    indirect_effects = np.array(indirect_effects)
    mean_indirect = np.mean(indirect_effects)
    ci_low, ci_high = np.percentile(indirect_effects, [2.5, 97.5])

    return mean_indirect, ci_low, ci_high

# ============================================================
# 8. SINGLE RUN
# ============================================================

def run_pipeline(seed):

    C = generate_connectivity(40, seed)
    df = generate_dataset(C)

    # ECM
    X_ecm = df[["Age", "Energy_Var"]].values
    y_ecm = df["Cog_ECM"].values
    score_ecm, p_ecm = permutation_test(X_ecm, y_ecm)

    # FEP
    X_fep = df[["Age", "Error_Var"]].values
    y_fep = df["Cog_FEP"].values
    score_fep, p_fep = permutation_test(X_fep, y_fep)

    # Mediation
    med_ecm_mean, med_ecm_low, med_ecm_high = mediation_bootstrap(
        df, "Energy_Var", "Cog_ECM"
    )

    med_fep_mean, med_fep_low, med_fep_high = mediation_bootstrap(
        df, "Error_Var", "Cog_FEP"
    )

    curvature_peak = regime_divergence(C)

    return {
        "ECM_R2": score_ecm,
        "ECM_p": p_ecm,
        "FEP_R2": score_fep,
        "FEP_p": p_fep,
        "ECM_med_mean": med_ecm_mean,
        "ECM_med_CI_low": med_ecm_low,
        "ECM_med_CI_high": med_ecm_high,
        "FEP_med_mean": med_fep_mean,
        "FEP_med_CI_low": med_fep_low,
        "FEP_med_CI_high": med_fep_high,
        "Regime_curvature_peak": curvature_peak
    }

# ============================================================
# 9. MULTI-SEED ROBUSTNESS
# ============================================================

if __name__ == "__main__":

    print("\n=== STRESSED Q1 ENERGY-CONSTRAINED PIPELINE ===\n")

    results = []

    for seed in range(20):
        print(f"Running seed {seed}")
        results.append(run_pipeline(seed))

    df_res = pd.DataFrame(results)

    print("\nROBUSTNESS SUMMARY (Mean Across Seeds)\n")
    print(df_res.mean())

    print("\nDetailed Summary:\n")
    print(df_res.describe())
