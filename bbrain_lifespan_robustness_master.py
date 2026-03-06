import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import pickle
import os

# ============================================================
# 1. NETWORK CONNECTIVITY
# ============================================================

def create_connectivity(N=40):
    C = np.random.rand(N, N)
    C = (C + C.T) / 2
    np.fill_diagonal(C, 0)

    eigvals = np.linalg.eigvals(C)
    maxeig = np.max(np.abs(eigvals))
    if maxeig == 0:
        maxeig = 1

    C = C / maxeig
    return C


# ============================================================
# 2. ENERGY CONSTRAINED MODEL (ECM)
# ============================================================

def simulate_ECM(C, T=200, dt=0.05, G=0.2,
                 alpha=1.0, beta=0.6, delta=0.05, noise=0.02):

    N = C.shape[0]

    omega = np.random.uniform(0.04, 0.07, N)
    z = np.random.randn(N) + 1j*np.random.randn(N)

    E = 1.0

    steps = int(T/dt)

    Z = np.zeros((steps, N), dtype=complex)
    E_series = np.zeros(steps)

    row_sums = np.sum(C, axis=1)

    for t in range(steps):

        neural_energy = np.mean(np.abs(z)**2)

        dE = alpha - beta*neural_energy - delta*E
        E += dE*dt
        E = max(E, 0.01)

        coupling = G*(C@z - row_sums*z)

        dz = (0.02 + 1j*omega - np.abs(z)**2)*z
        dz += coupling
        dz -= (1.0/E)*z

        dz += noise*(np.random.randn(N) + 1j*np.random.randn(N))

        z += dz*dt

        Z[t] = z
        E_series[t] = E

    return Z, E_series


# ============================================================
# 3. FREE ENERGY STYLE MODEL
# ============================================================

def simulate_FEP(C, T=200, dt=0.05, G=0.2,
                 precision=1.0, noise=0.02):

    N = C.shape[0]

    omega = np.random.uniform(0.04, 0.07, N)
    z = np.random.randn(N) + 1j*np.random.randn(N)

    steps = int(T/dt)

    Z = np.zeros((steps, N), dtype=complex)

    row_sums = np.sum(C, axis=1)

    for t in range(steps):

        prediction = C @ z
        pred_error = z - prediction

        coupling = G*(C@z - row_sums*z)

        dz = (0.02 + 1j*omega - np.abs(z)**2)*z
        dz += coupling
        dz -= precision * pred_error

        dz += noise*(np.random.randn(N) + 1j*np.random.randn(N))

        z += dz*dt

        Z[t] = z

    return Z


# ============================================================
# 4. NULL MODEL
# ============================================================

def simulate_null(N, T=200, dt=0.05, noise=0.02):

    steps = int(T/dt)

    Z = noise*(np.random.randn(steps, N) +
               1j*np.random.randn(steps, N))

    return Z


# ============================================================
# 5. COGNITION GENERATOR (INDEPENDENT)
# ============================================================

def generate_cognition(age, latent_trait):

    age_effect = 0.03 * age - 0.00035 * age**2
    noise = np.random.normal(0, 0.3)

    cognition = latent_trait + age_effect + noise

    return cognition


# ============================================================
# 6. METRIC COMPUTATION
# ============================================================

def compute_entropy_sync(Z):

    phases = np.angle(Z)

    R = np.abs(np.exp(1j*phases).mean(axis=1))

    hist = np.histogram(R, bins=40, density=True)[0] + 1e-8

    return entropy(hist), R


def compute_EECI(Z, E_series):

    H, _ = compute_entropy_sync(Z)

    energy_var = np.var(E_series)

    return H/(1 + energy_var)


def compute_FECI(Z):

    H, R = compute_entropy_sync(Z)

    pred_error = np.diff(R)

    return H/(1 + np.var(pred_error))


# ============================================================
# 7. LONGITUDINAL DATA GENERATION
# ============================================================

def generate_longitudinal_data_all(C,
                                   n_subjects=60,
                                   n_timepoints=6,
                                   age_min=10,
                                   age_max=80):

    rows = []

    ages = np.linspace(age_min, age_max, n_timepoints)

    N = C.shape[0]

    for subj in range(n_subjects):

        latent_trait = np.random.normal(0,1)

        subj_shift = np.random.normal(0,0.05)

        for age in ages:

            alpha_age = 1.2 - 0.005*age + subj_shift

            Z_ecm, E_series = simulate_ECM(C, alpha=alpha_age)
            EECI = compute_EECI(Z_ecm, E_series)

            Z_fep = simulate_FEP(C)
            FECI = compute_FECI(Z_fep)

            Z_null = simulate_null(N)
            NullCI = compute_FECI(Z_null)

            Cog = generate_cognition(age, latent_trait)

            rows.append({
                "Subject": subj,
                "Age": age,
                "EECI": EECI,
                "FECI": FECI,
                "NullCI": NullCI,
                "Cog": Cog
            })

    return pd.DataFrame(rows)


# ============================================================
# 8. CROSS VALIDATED PREDICTION
# ============================================================

def cross_validated_r2_multi(df, predictors):

    X = df[predictors].values
    y = df["Cog"].values

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    r2s = []

    for train_idx, test_idx in kf.split(X):

        model = LinearRegression()

        model.fit(X[train_idx], y[train_idx])

        pred = model.predict(X[test_idx])

        r = np.corrcoef(y[test_idx], pred)[0,1]

        if np.isnan(r):
            r = 0

        r2s.append(r**2)

    return np.mean(r2s)


def permutation_test_multi(df, predictor="EECI", n_perm=100):

    baseline = cross_validated_r2_multi(df, [predictor])

    perm_r2 = []

    for _ in range(n_perm):

        df_perm = df.copy()

        df_perm[predictor] = np.random.permutation(df_perm[predictor].values)

        perm_r2.append(
            cross_validated_r2_multi(df_perm, [predictor])
        )

    p_value = np.mean(np.array(perm_r2) >= baseline)

    return baseline, perm_r2, p_value


# ============================================================
# 9. MULTI-SEED ROBUSTNESS
# ============================================================

def run_robustness_all(n_seeds=10, save_dir="Model_Results"):

    os.makedirs(save_dir, exist_ok=True)

    all_results = []

    for seed in range(n_seeds):

        print("Running seed", seed)

        np.random.seed(seed)

        C = create_connectivity()

        df = generate_longitudinal_data_all(C)

        results = {}

        for model in ["EECI","FECI","NullCI"]:

            r2 = cross_validated_r2_multi(df, [model])

            baseline, perm_r2, pval = permutation_test_multi(df, predictor=model)

            results[model+"_R2"] = r2
            results[model+"_p"] = pval

        results["Seed"] = seed

        all_results.append(results)

        with open(os.path.join(save_dir,
                 f"seed_{seed}_workspace.pkl"),"wb") as f:

            pickle.dump(df,f)

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        os.path.join(save_dir,"robustness_summary.csv"),
        index=False
    )

    return results_df


# ============================================================
# 10. PUBLICATION FIGURES
# ============================================================

def plot_results_all(results_df, save_dir="Model_Results"):

    sns.set(style="whitegrid",context="talk")

    fig, axes = plt.subplots(1,3,figsize=(18,6))

    sns.histplot(results_df["EECI_R2"], kde=True, ax=axes[0])
    sns.histplot(results_df["FECI_R2"], kde=True, ax=axes[0])
    sns.histplot(results_df["NullCI_R2"], kde=True, ax=axes[0])

    axes[0].set_title("Cross Validated R²")
    axes[0].legend(["ECM","FEP","Null"])

    sns.histplot(results_df["EECI_p"], kde=True, ax=axes[1])
    sns.histplot(results_df["FECI_p"], kde=True, ax=axes[1])
    sns.histplot(results_df["NullCI_p"], kde=True, ax=axes[1])

    axes[1].set_title("Permutation p-values")
    axes[1].legend(["ECM","FEP","Null"])

    axes[2].plot(results_df.index, results_df["EECI_R2"], 'o-', label="ECM")
    axes[2].plot(results_df.index, results_df["FECI_R2"], 'o-', label="FEP")
    axes[2].plot(results_df.index, results_df["NullCI_R2"], 'o-', label="Null")

    axes[2].set_title("R² per Seed")
    axes[2].set_xlabel("Seed")
    axes[2].set_ylabel("R²")
    axes[2].legend()

    plt.tight_layout()

    fig_path = os.path.join(save_dir,"Model_Figures.png")

    plt.savefig(fig_path, dpi=300)

    plt.show()

    print("Figures saved to", fig_path)


# ============================================================
# 11. MASTER EXECUTION
# ============================================================

if __name__ == "__main__":

    print("=== MULTI MODEL COMPARISON PIPELINE ===")

    results = run_robustness_all(n_seeds=10)

    print("\nROBUSTNESS SUMMARY")
    print(results.mean())

    plot_results_all(results)
