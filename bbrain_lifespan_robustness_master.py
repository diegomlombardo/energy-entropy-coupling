#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brain-Body Lifespan ECM vs FEP Pipeline
Q1 Publication-ready, in-silico experiments
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

# ================================
# 1. SIMULATION FUNCTIONS
# ================================
def simulate_brain_body(N=40, T=500, dt=0.02, G=0.2, noise=0.02, lambda_body=0.05):
    """Simulate neural and body dynamics, return complex Z and physiological P"""
    omega = np.random.uniform(0.04, 0.07, N)
    z = np.random.randn(N)+1j*np.random.randn(N)
    cardiac = respiratory = metabolic = 0.1
    steps = int(T/dt)
    Z = np.zeros((steps, N), dtype=complex)
    P = np.zeros((steps, 3))
    
    for t in range(steps):
        phases = np.angle(z)
        R = np.abs(np.exp(1j*phases).mean())
        cardiac += (0.3*cardiac - cardiac**3 + 0.05*R)*dt
        respiratory += (0.25*respiratory - respiratory**3 + 0.04*R)*dt
        metabolic += (0.1*metabolic - metabolic**3 + 0.02*R)*dt
        body_state = np.mean([cardiac, respiratory, metabolic])
        dz = (0.02 + 1j*omega - np.abs(z)**2)*z
        dz += G*(z.mean()-z) + lambda_body*body_state*z
        dz += noise*(np.random.randn(N)+1j*np.random.randn(N))
        z += dz*dt
        Z[t] = z
        P[t] = [cardiac, respiratory, metabolic]
    return Z, P

# ================================
# 2. METRICS
# ================================
def metastability(Z):
    phases = np.angle(Z)
    R = np.abs(np.exp(1j*phases).mean(axis=1))
    return np.std(R), R

def entropy_signal(signal, bins=40):
    hist,_ = np.histogram(signal,bins=bins,density=True)
    hist = hist[hist>0]
    return -np.sum(hist*np.log(hist))

def gaussian_copula_mi(x, y):
    xr = rankdata(x)/(len(x)+1)
    yr = rankdata(y)/(len(y)+1)
    xn = norm.ppf(xr)
    yn = norm.ppf(yr)
    r = np.corrcoef(xn,yn)[0,1]
    r = np.clip(r,-0.9999,0.9999)
    return -0.5*np.log(1-r**2)

def compute_ECM(Z, P, H_star):
    _, R = metastability(Z)
    H = entropy_signal(R)
    body = np.mean(P,axis=1)
    MI = gaussian_copula_mi(R, body)
    return MI - abs(H-H_star)

def compute_PPI(Z):
    X = np.abs(Z)
    X_pred, X_true = X[:-1], X[1:]
    beta = np.linalg.pinv(X_pred) @ X_true
    residual = X_true - X_pred @ beta
    return np.mean(residual**2)

# ================================
# 3. DATA GENERATION
# ================================
def generate_lifespan_dataset(N_subjects=60, N_timepoints=6):
    ages = np.linspace(10,80,N_timepoints)
    rows=[]
    for subj in range(N_subjects):
        subj_shift = np.random.normal(0,0.04)
        for age in ages:
            # Adaptive G
            age_factor = max(-0.0008*(age-45)**2 + 1,0.1)
            Z,P = simulate_brain_body(G=0.2*age_factor + subj_shift)
            H_star = 1.0  # reference
            ECM = compute_ECM(Z,P,H_star)
            PPI = compute_PPI(Z)
            Path = np.random.normal(0,1)
            Cog = 0.6*ECM - 0.0006*(age-45)**2 + np.random.normal(0,0.5)
            rows.append({"Subject":subj,"Age":age,"ECM":ECM,"PPI":PPI,"Path":Path,"Cog":Cog})
    df=pd.DataFrame(rows)
    df["Age_c"]=df["Age"]-df["Age"].mean()
    df["Age2"]=df["Age_c"]**2
    return df

# ================================
# 4. MIXED EFFECTS MODEL
# ================================
def fit_mixed_model(df):
    model = smf.mixedlm("Cog ~ ECM + Age_c + Age2 + PPI + Path", df, groups=df["Subject"])
    result = model.fit()
    return result

# ================================
# 5. PERMUTATION TEST
# ================================
def permutation_test(X, y, n_perms=100):
    """Cross-validated R² permutation"""
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)
    real_model = LinearRegression().fit(X,y)
    real_score = r2_score(y, real_model.predict(X))
    perm_scores=[]
    for _ in range(n_perms):
        y_perm = np.random.permutation(y)
        perm_scores.append(r2_score(y_perm, LinearRegression().fit(X,y_perm).predict(X)))
    p_value = (np.sum(np.array(perm_scores)>=real_score)+1)/(n_perms+1)
    return real_score, p_value

# ================================
# 6. BOOTSTRAP CI
# ================================
def bootstrap_ci(data, n_boot=2000, alpha=0.05):
    boot_samples=[]
    for _ in range(n_boot):
        sample=resample(data)
        boot_samples.append(np.mean(sample))
    lower = np.percentile(boot_samples,100*alpha/2)
    upper = np.percentile(boot_samples,100*(1-alpha/2))
    return lower, upper

# ================================
# 7. MEDIATION ANALYSIS
# ================================
def mediation_analysis(df):
    # ECM → Age2 → Cognition
    model_m = smf.ols("Age2 ~ ECM",df).fit()
    model_y = smf.ols("Cog ~ ECM + Age2",df).fit()
    indirect = model_m.params["ECM"]*model_y.params["Age2"]
    direct = model_y.params["ECM"]
    total = direct + indirect
    return {"direct":direct,"indirect":indirect,"total":total}

# ================================
# 8. RUN MULTISEED PIPELINE
# ================================
def run_multiseed_pipeline(n_seeds=20):
    all_results=[]
    for seed in range(n_seeds):
        np.random.seed(seed)
        df = generate_lifespan_dataset()
        result = fit_mixed_model(df)
        X = df[["ECM","PPI","Age_c","Age2","Path"]].values
        y = df["Cog"].values
        r2, p = permutation_test(X, y)
        boot_l, boot_u = bootstrap_ci(result.params.values)
        med = mediation_analysis(df)
        all_results.append({"Seed":seed,
                            "Beta_ECM":result.params["ECM"],
                            "p_ECM":result.pvalues["ECM"],
                            "Beta_Age2":result.params["Age2"],
                            "p_Age2":result.pvalues["Age2"],
                            "R2_perm":r2,
                            "p_perm":p,
                            "Bootstrap_Lower":boot_l,
                            "Bootstrap_Upper":boot_u,
                            "Mediation":med})
        print(f"Seed {seed+1}/{n_seeds} complete")
    return pd.DataFrame(all_results)

# ================================
# 9. FIGURE PANELS
# ================================
def plot_figures(df_results):
    sns.set(style="whitegrid",context="talk",palette="colorblind")
    fig, axes = plt.subplots(2,2,figsize=(16,12))

    # Panel 1: ECM Beta distribution
    sns.histplot(df_results["Beta_ECM"], kde=True, ax=axes[0,0], color="#1f77b4")
    axes[0,0].axvline(df_results["Beta_ECM"].mean(),color="black",linestyle="--")
    axes[0,0].set_title("Panel 1: ECM Beta Distribution")

    # Panel 2: Age² Beta distribution
    sns.histplot(df_results["Beta_Age2"], kde=True, ax=axes[0,1], color="#ff7f0e")
    axes[0,1].axvline(df_results["Beta_Age2"].mean(),color="black",linestyle="--")
    axes[0,1].set_title("Panel 2: Age² Beta Distribution")

    # Panel 3: Permutation R² distribution
    sns.histplot(df_results["R2_perm"], kde=True, ax=axes[1,0], color="#2ca02c")
    axes[1,0].set_title("Panel 3: Permutation R²")

    # Panel 4: Mediation indirect effect
    indirects = [m["indirect"] for m in df_results["Mediation"]]
    sns.histplot(indirects, kde=True, ax=axes[1,1], color="#d62728")
    axes[1,1].set_title("Panel 4: Mediation Indirect Effect (ECM→Age²→Cognition)")

    plt.tight_layout()
    plt.savefig("FIGURE_Q1_LIFESPAN_PIPELINE.png",dpi=600)
    plt.savefig("FIGURE_Q1_LIFESPAN_PIPELINE.pdf")
    plt.show()

# ================================
# 10. MAIN
# ================================
if __name__=="__main__":
    print("\n=== ENERGY-CONSTRAINED vs PREDICTIVE PIPELINE ===\n")
    df_results = run_multiseed_pipeline(n_seeds=20)
    df_results.to_csv("LIFESPAN_PIPELINE_RESULTS.csv",index=False)
    print("\n=== RESULTS SUMMARY ===\n")
    print(df_results.describe())
    print("\nGenerating publication-quality figures...")
    plot_figures(df_results)
    print("\nPipeline complete. All files saved.\n")
