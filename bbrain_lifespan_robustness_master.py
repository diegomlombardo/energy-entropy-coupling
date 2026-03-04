# ============================================================
# Q1 POWER-CALIBRATED CONSTRUCT VALIDATION
# ============================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.utils import resample

# ============================================================
# 1. CONNECTIVITY
# ============================================================

def create_connectivity(N=40):
    C = np.random.rand(N, N)
    C = (C + C.T) / 2
    np.fill_diagonal(C, 0)
    C = C / np.max(np.abs(np.linalg.eigvals(C)))
    return C


# ============================================================
# 2. MEASUREMENT LAYER
# ============================================================

def simulate_EECI(C, age):

    N = C.shape[0]
    z = np.random.randn(N)
    energy = 1.0
    alpha = 1.2 - 0.005 * age

    for _ in range(250):
        neural_energy = np.mean(z**2)
        energy += (alpha - 0.6*neural_energy - 0.05*energy)*0.05
        energy = np.clip(energy, 0.01, 5)
        z += 0.1*(C @ z - z) - (1/energy)*z + np.random.randn(N)*0.02

    return np.var(z)


def simulate_PPI(C, age):

    N = C.shape[0]
    z = np.random.randn(N)

    for _ in range(250):
        prediction = C @ z
        pred_error = z - prediction
        z += -0.8*pred_error + np.random.randn(N)*0.02

    return np.var(pred_error)


# ============================================================
# 3. GENERATIVE COGNITION WITH EXPLICIT SNR
# ============================================================

def generate_cognition(world, eeci, ppi, snr=3.0):

    if world == "energy":
        signal = eeci
    elif world == "predictive":
        signal = ppi
    elif world == "mixed":
        signal = 0.5*eeci + 0.5*ppi
    else:
        raise ValueError("Invalid world")

    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    noise_var = np.var(signal) / snr
    noise = np.random.randn(*signal.shape) * np.sqrt(noise_var)

    return signal + noise


# ============================================================
# 4. DATASET GENERATION
# ============================================================

def generate_dataset(world, n_subjects=120):

    C = create_connectivity()

    ages = np.random.uniform(10, 80, n_subjects)

    eeci = np.array([simulate_EECI(C, age) for age in ages])
    ppi  = np.array([simulate_PPI(C, age) for age in ages])

    cog = generate_cognition(world, eeci, ppi)

    return pd.DataFrame({
        "Age": ages,
        "EECI": eeci,
        "PPI": ppi,
        "Cog": cog
    })


# ============================================================
# 5. STATISTICS (PROPER NESTED MODEL)
# ============================================================

def compute_stats(df):

    scaler = StandardScaler()
    X_full = scaler.fit_transform(df[["EECI","PPI","Age"]])
    y = StandardScaler().fit_transform(df[["Cog"]]).flatten()

    model_full = LinearRegression().fit(X_full, y)
    r2_full = r2_score(y, model_full.predict(X_full))
    beta_eeci = model_full.coef_[0]

    # Reduced model (remove EECI)
    X_reduced = X_full[:,1:]
    model_reduced = LinearRegression().fit(X_reduced, y)
    r2_reduced = r2_score(y, model_reduced.predict(X_reduced))

    delta_r2 = r2_full - r2_reduced
    partial_r2 = delta_r2 / (1 - r2_reduced + 1e-8)

    return beta_eeci, delta_r2, partial_r2


# ============================================================
# 6. PERMUTATION TEST
# ============================================================

def permutation_test(df, n_perm=300):

    beta_real, _, _ = compute_stats(df)

    perm_betas = []

    for _ in range(n_perm):
        df_perm = df.copy()
        df_perm["EECI"] = np.random.permutation(df_perm["EECI"])
        beta_perm, _, _ = compute_stats(df_perm)
        perm_betas.append(beta_perm)

    p = np.mean(np.abs(perm_betas) >= np.abs(beta_real))
    return p


# ============================================================
# 7. ROBUSTNESS ACROSS SEEDS
# ============================================================

def run_world(world, n_runs=50):

    betas = []
    deltas = []
    partials = []
    pvals = []

    for seed in range(n_runs):
        np.random.seed(seed)

        df = generate_dataset(world)

        beta, delta_r2, partial_r2 = compute_stats(df)
        pval = permutation_test(df)

        betas.append(beta)
        deltas.append(delta_r2)
        partials.append(partial_r2)
        pvals.append(pval)

    return {
        "beta_mean": np.mean(betas),
        "beta_CI": np.percentile(betas,[2.5,97.5]),
        "deltaR2_mean": np.mean(deltas),
        "partialR2_mean": np.mean(partials),
        "significance_rate": np.mean(np.array(pvals)<0.05)
    }


# ============================================================
# 8. MASTER EXECUTION
# ============================================================

if __name__ == "__main__":

    print("\n=== POWER-CALIBRATED WORLD VALIDATION ===\n")

    for world in ["energy","predictive","mixed"]:
        print("\nWORLD:", world.upper())
        results = run_world(world)
        for k,v in results.items():
            print(k,":",v)
