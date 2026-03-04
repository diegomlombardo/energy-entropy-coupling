# ============================================================
# Q1 ENERGY-CONSTRAINED COGNITION PIPELINE
# - Cognition = perturbation recovery only
# - Linear metabolic decline only
# - Explicit energy reservoir
# - No entropy targets
# - No imposed quadratic lifespan
# - ECM vs FEP competition
# - 10-fold CV
# - Permutation inference
# - Multi-seed robustness
# - Regime divergence
# - Mediation analysis
# ============================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.utils import resample
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# ============================================================
# 1. CONNECTIVITY
# ============================================================

def generate_connectivity(N, seed):
    np.random.seed(seed)
    C = np.random.rand(N,N)
    C = (C + C.T)/2
    np.fill_diagonal(C,0)
    C = C / np.max(np.abs(np.linalg.eigvals(C)))
    return C

# ============================================================
# 2. ECM SIMULATION (ENERGY EXPLICIT)
# ============================================================

def simulate_ECM(C, age,
                 T=300, dt=0.02,
                 alpha0=1.2,
                 metabolic_decline=0.005,
                 beta=0.6,
                 delta=0.05,
                 kappa=1.0,
                 noise=0.02):

    N = C.shape[0]
    omega = np.random.uniform(0.04, 0.07, N)
    z = np.random.randn(N) + 1j*np.random.randn(N)
    E = 1.0

    alpha = alpha0 - metabolic_decline * age  # strictly linear decline

    timepoints = int(T/dt)
    row_sums = np.sum(C, axis=1)

    recovery_times = []
    energy_series = []

    perturb_time = int(0.4*timepoints)
    threshold = 0.9

    for t in range(timepoints):

        neural_energy = np.mean(np.abs(z)**2)
        dE = alpha - beta*neural_energy - delta*E
        E += dE*dt
        E = max(E, 0.01)

        coupling = 0.2*(C @ z - row_sums*z)
        dz = (0.02 + 1j*omega - np.abs(z)**2)*z
        dz += coupling
        dz += -kappa*(1.0/E)*z
        dz += noise*(np.random.randn(N)+1j*np.random.randn(N))
        z += dz*dt

        if t == perturb_time:
            z *= 0.2  # perturbation

        if t > perturb_time:
            if np.mean(np.abs(z)) > threshold:
                recovery_times.append(t - perturb_time)

        energy_series.append(E)

    recovery = np.mean(recovery_times) if len(recovery_times)>0 else timepoints
    energy_var = np.var(energy_series)

    return recovery, energy_var

# ============================================================
# 3. FEP SIMULATION (NO ENERGY)
# ============================================================

def simulate_FEP(C, age,
                 T=300, dt=0.02,
                 precision=1.0,
                 noise=0.02):

    N = C.shape[0]
    omega = np.random.uniform(0.04, 0.07, N)
    z = np.random.randn(N) + 1j*np.random.randn(N)

    timepoints = int(T/dt)
    row_sums = np.sum(C, axis=1)

    recovery_times = []
    error_series = []

    perturb_time = int(0.4*timepoints)
    threshold = 0.9

    for t in range(timepoints):

        prediction = C @ z
        prediction_error = z - prediction
        error_series.append(np.mean(np.abs(prediction_error)))

        coupling = 0.2*(C @ z - row_sums*z)
        dz = (0.02 + 1j*omega - np.abs(z)**2)*z
        dz += coupling
        dz += -precision*prediction_error
        dz += noise*(np.random.randn(N)+1j*np.random.randn(N))
        z += dz*dt

        if t == perturb_time:
            z *= 0.2

        if t > perturb_time:
            if np.mean(np.abs(z)) > threshold:
                recovery_times.append(t - perturb_time)

    recovery = np.mean(recovery_times) if len(recovery_times)>0 else timepoints
    error_var = np.var(error_series)

    return recovery, error_var

# ============================================================
# 4. GENERATE LIFESPAN DATA (NO IMPOSED COGNITION)
# ============================================================

def generate_dataset(C, n_subjects=80):

    ages = np.linspace(10,80,n_subjects)
    rows = []

    for age in ages:

        rec_ecm, energy_var = simulate_ECM(C, age)
        rec_fep, error_var = simulate_FEP(C, age)

        # Cognition = inverse recovery time
        cognition_ecm = 1.0 / rec_ecm
        cognition_fep = 1.0 / rec_fep

        rows.append({
            "Age":age,
            "Cog_ECM":cognition_ecm,
            "Cog_FEP":cognition_fep,
            "Energy_Var":energy_var,
            "Error_Var":error_var
        })

    return pd.DataFrame(rows)

# ============================================================
# 5. 10-FOLD CROSS VALIDATION
# ============================================================

def cross_validate_model(X, y):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = []

    for train, test in kf.split(X):
        model = LinearRegression().fit(X[train], y[train])
        pred = model.predict(X[test])
        scores.append(r2_score(y[test], pred))

    return np.mean(scores)

# ============================================================
# 6. PERMUTATION TEST
# ============================================================

def permutation_test(X, y, n_perm=1000):
    real_score = cross_validate_model(X,y)
    perm_scores = []

    for _ in range(n_perm):
        y_perm = np.random.permutation(y)
        perm_scores.append(cross_validate_model(X,y_perm))

    p_value = np.mean(np.array(perm_scores) >= real_score)
    return real_score, p_value

# ============================================================
# 7. REGIME DIVERGENCE
# ============================================================

def regime_divergence(C):

    kappas = np.linspace(0.2,3.0,20)
    sync_vals = []

    for k in kappas:
        rec,_ = simulate_ECM(C, age=40, kappa=k)
        sync_vals.append(1.0/rec)

    second_deriv = np.gradient(np.gradient(sync_vals))
    return kappas, sync_vals, second_deriv

# ============================================================
# 8. MEDIATION (BOOTSTRAP)
# ============================================================

def mediation_bootstrap(df, mediator_col, cog_col, n_boot=1000):

    indirect = []

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

        indirect.append(a*b)

    ci = np.percentile(indirect,[2.5,97.5])
    return np.mean(indirect), ci

# ============================================================
# 9. MASTER PIPELINE
# ============================================================

def run_pipeline(seed=0):

    C = generate_connectivity(40,seed)
    df = generate_dataset(C)

    # ECM model
    X_ecm = df[["Age","Energy_Var"]].values
    y_ecm = df["Cog_ECM"].values

    score_ecm, p_ecm = permutation_test(X_ecm,y_ecm)

    # FEP model
    X_fep = df[["Age","Error_Var"]].values
    y_fep = df["Cog_FEP"].values

    score_fep, p_fep = permutation_test(X_fep,y_fep)

    med_ecm = mediation_bootstrap(df,"Energy_Var","Cog_ECM")
    med_fep = mediation_bootstrap(df,"Error_Var","Cog_FEP")

    kappas, sync, curv = regime_divergence(C)

    return {
        "ECM_R2":score_ecm,
        "ECM_p":p_ecm,
        "FEP_R2":score_fep,
        "FEP_p":p_fep,
        "ECM_mediation":med_ecm,
        "FEP_mediation":med_fep,
        "Regime_curvature_peak":np.max(np.abs(curv))
    }

# ============================================================
# 10. MULTI-SEED ROBUSTNESS
# ============================================================

if __name__=="__main__":

    results = []

    for seed in range(20):
        print(f"Running seed {seed}")
        results.append(run_pipeline(seed))

    df_res = pd.DataFrame(results)
    print("\nROBUSTNESS SUMMARY\n")
    print(df_res.mean())
