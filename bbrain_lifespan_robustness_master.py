import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import entropy, rankdata, norm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. BRAIN–BODY SIMULATOR
# ============================================================
def simulate_brain_body(C, T=500, dt=0.02, G=0.2,
                        noise=0.02, lambda_body=0.05):

    N = C.shape[0]
    omega = np.random.uniform(0.04, 0.07, N)
    z = np.random.randn(N) + 1j*np.random.randn(N)

    cardiac = respiratory = metabolic = 0.1
    timepoints = int(T/dt)
    Z = np.zeros((timepoints, N), dtype=complex)
    P = np.zeros((timepoints, 3))
    row_sums = np.sum(C, axis=1)

    for t in range(timepoints):
        phases = np.angle(z)
        R = np.abs(np.exp(1j*phases).mean())
        cardiac += (0.3*cardiac - cardiac**3 + 0.05*R)*dt
        respiratory += (0.25*respiratory - respiratory**3 + 0.04*R)*dt
        metabolic += (0.1*metabolic - metabolic**3 + 0.02*R)*dt
        body_state = np.mean([cardiac, respiratory, metabolic])

        coupling = G * (C @ z - row_sums*z)
        dz = (0.02 + 1j*omega - np.abs(z)**2)*z
        dz += coupling + lambda_body * body_state * z
        dz += noise*(np.random.randn(N)+1j*np.random.randn(N))
        z += dz*dt

        Z[t] = z
        P[t] = [cardiac, respiratory, metabolic]

    return Z, P

# ============================================================
# 2. METRICS
# ============================================================
def compute_metastability(Z):
    phases = np.angle(Z)
    R = np.abs(np.exp(1j*phases).mean(axis=1))
    return np.std(R), R

def compute_entropy(signal, bins=40):
    hist, _ = np.histogram(signal, bins=bins, density=True)
    hist = hist[hist>0]
    return entropy(hist)

def gaussian_copula_mi(x, y):
    xr = rankdata(x)/(len(x)+1)
    yr = rankdata(y)/(len(y)+1)
    xn = norm.ppf(xr)
    yn = norm.ppf(yr)
    r = np.corrcoef(xn, yn)[0,1]
    r = np.clip(r, -0.9999, 0.9999)
    return -0.5*np.log(1-r**2)

def compute_eeci(Z, P, H_star):
    _, R = compute_metastability(Z)
    H = compute_entropy(R)
    body = np.mean(P, axis=1)
    MI = gaussian_copula_mi(R, body)
    return MI - abs(H - H_star)

def compute_ppi(Z):
    X = np.abs(Z)
    X_pred, X_true = X[:-1], X[1:]
    beta = np.linalg.pinv(X_pred) @ X_true
    residual = X_true - X_pred @ beta
    return np.mean(residual**2)

# ============================================================
# 3. OPTIMAL COUPLING
# ============================================================
def find_optimal_G(C):
    G_values = np.linspace(0.05, 0.4, 6)
    best_score = -np.inf
    for G in G_values:
        Z, P = simulate_brain_body(C, G=G)
        _, R = compute_metastability(Z)
        H = compute_entropy(R)
        E = np.mean(np.abs(Z)**2)
        score = H - 0.5*E
        if score > best_score:
            best_score = score
            best_G = G
            best_H = H
    return best_G, best_H

# ============================================================
# 4. LONGITUDINAL COHORT
# ============================================================
def generate_lifespan_data(C, n_subjects=60, n_timepoints=6):
    ages = np.linspace(10,80,n_timepoints)
    G_opt, H_star = find_optimal_G(C)
    rows = []

    for subj in range(n_subjects):
        subj_shift = np.random.normal(0,0.04)
        for age in ages:
            age_factor = -0.0008*(age-45)**2 + 1
            Z,P = simulate_brain_body(C, G=G_opt*age_factor + subj_shift)
            EECI = compute_eeci(Z,P,H_star)
            PPI = compute_ppi(Z)
            Path = np.random.normal(0,1)
            Cog = 0.6*EECI - 0.0006*(age-45)**2 + np.random.normal(0,0.5)
            rows.append({"Subject":subj,"Age":age,"EECI":EECI,"PPI":PPI,"Path":Path,"Cog":Cog})

    df = pd.DataFrame(rows)
    df["Age_c"] = df["Age"] - df["Age"].mean()
    df["Age2"] = df["Age_c"]**2
    return df

# ============================================================
# 5. MIXED MODEL
# ============================================================
def fit_mixed_model(df):
    model = smf.mixedlm("Cog ~ EECI + Age_c + Age2 + PPI + Path",
                        df, groups=df["Subject"])
    result = model.fit()
    return result

# ============================================================
# 6. ROBUSTNESS PIPELINE
# ============================================================
def run_longitudinal_robustness(n_runs=50):
    results = []
    for seed in range(n_runs):
        np.random.seed(seed)
        N = 40
        C = np.random.rand(N,N)
        C = (C + C.T)/2
        np.fill_diagonal(C,0)
        C = C / np.max(np.abs(np.linalg.eigvals(C)))

        df = generate_lifespan_data(C)
        try:
            model = fit_mixed_model(df)
            results.append({
                "Seed":seed,
                "Beta_EECI":model.params["EECI"],
                "p_EECI":model.pvalues["EECI"],
                "Beta_Age2":model.params["Age2"],
                "p_Age2":model.pvalues["Age2"]
            })
        except:
            continue
        print(f"Run {seed+1}/{n_runs} complete")
    return pd.DataFrame(results)

# ============================================================
# 7. PUBLICATION FIGURE
# ============================================================
def generate_publication_figure_longitudinal():
    df = pd.read_csv("LONGITUDINAL_FULL_DISTRIBUTIONS.csv")
    summary = pd.read_csv("LONGITUDINAL_ROBUSTNESS_SUMMARY.csv")

    sns.set(style="whitegrid", context="talk", palette="colorblind")
    fig, axes = plt.subplots(1,3,figsize=(18,6))

    # Panel A: predicted inverted-U trajectory
    ages = np.linspace(10,80,200)
    beta_age2 = df["Beta_Age2"].mean()
    intercept = 0
    cognition_pred = intercept + beta_age2*(ages-45)**2
    axes[0].plot(ages,cognition_pred,color="#1f77b4",lw=3)
    axes[0].set_title("A: Predicted Cognition Trajectory")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Predicted Cognition")

    # Panel B: EECI robustness distribution
    sns.histplot(df["Beta_EECI"], kde=True, ax=axes[1], color="#ff7f0e", bins=20)
    axes[1].axvline(df["Beta_EECI"].mean(), color="black", linestyle="--")
    axes[1].set_title("B: EECI Robustness Distribution")
    axes[1].set_xlabel("Beta EECI")
    axes[1].set_ylabel("Frequency")

    # Panel C: Age² robustness distribution
    sns.histplot(df["Beta_Age2"], kde=True, ax=axes[2], color="#2ca02c", bins=20)
    axes[2].axvline(df["Beta_Age2"].mean(), color="black", linestyle="--")
    axes[2].set_title("C: Age² Robustness Distribution")
    axes[2].set_xlabel("Beta Age²")
    axes[2].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("FIGURE_Longitudinal_Robustness.png", dpi=600)
    plt.savefig("FIGURE_Longitudinal_Robustness.pdf")
    plt.show()

# ============================================================
# 8. MAIN MASTER PIPELINE
# ============================================================
if __name__=="__main__":
    print("\n=== LONGITUDINAL LIFESPAN ROBUSTNESS PIPELINE (50 runs) ===\n")
    robustness_df = run_longitudinal_robustness(n_runs=50)

    summary = pd.DataFrame({
        "Mean_Beta_EECI":[robustness_df["Beta_EECI"].mean()],
        "SD_Beta_EECI":[robustness_df["Beta_EECI"].std()],
        "Prop_sig_EECI":[(robustness_df["p_EECI"]<0.05).mean()],
        "Mean_Beta_Age2":[robustness_df["Beta_Age2"].mean()],
        "SD_Beta_Age2":[robustness_df["Beta_Age2"].std()],
        "Prop_sig_Age2":[(robustness_df["p_Age2"]<0.05).mean()]
    })

    robustness_df.to_csv("LONGITUDINAL_FULL_DISTRIBUTIONS.csv", index=False)
    summary.to_csv("LONGITUDINAL_ROBUSTNESS_SUMMARY.csv", index=False)
    with open("LONGITUDINAL_ROBUSTNESS_WORKSPACE.pkl","wb") as f:
        pickle.dump({"full":robustness_df,"summary":summary}, f)

    print("\nROBUSTNESS SUMMARY:")
    print(summary)

    print("\nGenerating publication-quality figure...")
    generate_publication_figure_longitudinal()

    print("\nFiles saved:")
    print(" - LONGITUDINAL_FULL_DISTRIBUTIONS.csv")
    print(" - LONGITUDINAL_ROBUSTNESS_SUMMARY.csv")
    print(" - LONGITUDINAL_ROBUSTNESS_WORKSPACE.pkl")
    print(" - FIGURE_Longitudinal_Robustness.png / .pdf")
    print("\nPipeline complete.\n")
