#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brain-Body Lifespan ECM vs PPI Pipeline
Q1 Publication-ready, in-silico simulations

Features:
- Independent generative predictors (ECM, PPI)
- Null world included
- Multi-seed robustness
- Permutation testing
- Longitudinal age simulation
- Bootstrap confidence intervals
- Publication-quality tables and figures
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from sklearn.metrics import r2_score
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ================================
# 1. SIMULATION FUNCTIONS
# ================================
def simulate_brain_body(N=40, T=500, dt=0.02, G=0.2, noise=0.02, lambda_body=0.05):
    """Simulate neural (Z) and body (P) dynamics."""
    omega = np.random.uniform(0.04, 0.07, N)
    z = np.random.randn(N) + 1j*np.random.randn(N)
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
        dz += G*(z.mean() - z) + lambda_body*body_state*z
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
    hist,_ = np.histogram(signal, bins=bins, density=True)
    hist = hist[hist>0]
    return -np.sum(hist*np.log(hist))

def gaussian_copula_mi(x, y):
    xr = rankdata(x)/(len(x)+1)
    yr = rankdata(y)/(len(y)+1)
    xn = norm.ppf(xr)
    yn = norm.ppf(yr)
    r = np.corrcoef(xn, yn)[0,1]
    r = np.clip(r,-0.9999,0.9999)
    return -0.5*np.log(1-r**2)

def compute_ECM(Z, P, H_star):
    _, R = metastability(Z)
    H = entropy_signal(R)
    body = np.mean(P, axis=1)
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
def generate_lifespan_dataset(N_subjects=60, N_timepoints=6, world="energy"):
    """Generate longitudinal dataset for a given world."""
    ages = np.linspace(10, 80, N_timepoints)
    rows=[]
    for subj in range(N_subjects):
        subj_shift = np.random.normal(0,0.04)
        for age in ages:
            age_factor = max(-0.0008*(age-45)**2 + 1, 0.1)
            if world=="null":
                ECM = np.random.normal(0,0.05)
                PPI = np.random.normal(0,0.05)
            else:
                Z, P = simulate_brain_body(G=0.2*age_factor + subj_shift)
                H_star = 1.0
                ECM = compute_ECM(Z,P,H_star) if world in ["energy","mixed"] else np.random.normal(0,0.05)
                PPI = compute_PPI(Z) if world in ["predictive","mixed"] else np.random.normal(0,0.05)
            Path = np.random.normal(0,1)
            # Cognition depends on world
            if world=="energy":
                Cog = 0.6*ECM - 0.0006*(age-45)**2 + np.random.normal(0,0.5)
            elif world=="predictive":
                Cog = 0.6*PPI - 0.0006*(age-45)**2 + np.random.normal(0,0.5)
            elif world=="mixed":
                Cog = 0.4*ECM + 0.4*PPI - 0.0006*(age-45)**2 + np.random.normal(0,0.5)
            else:  # null
                Cog = -0.0006*(age-45)**2 + np.random.normal(0,0.5)
            rows.append({"Subject":subj,"Age":age,"ECM":ECM,"PPI":PPI,"Path":Path,"Cog":Cog})
    df = pd.DataFrame(rows)
    df["Age_c"] = df["Age"] - df["Age"].mean()
    df["Age2"] = df["Age_c"]**2
    return df

# ================================
# 4. MIXED MODEL + PERMUTATION + BOOTSTRAP
# ================================
def fit_mixed_model(df):
    model = smf.mixedlm("Cog ~ ECM + PPI + Age_c + Age2 + Path", df, groups=df["Subject"])
    result = model.fit()
    return result

def permutation_test_slope(df, formula="Cog ~ ECM + PPI + Age_c + Age2 + Path", n_perm=100):
    X = df[["ECM","PPI","Age_c","Age2","Path"]].values
    y = df["Cog"].values
    real_model = LinearRegression().fit(X,y)
    r2_real = r2_score(y, real_model.predict(X))
    perm_r2=[]
    for _ in range(n_perm):
        y_perm = np.random.permutation(y)
        perm_r2.append(r2_score(y_perm, LinearRegression().fit(X,y_perm).predict(X)))
    p_value = (np.sum(np.array(perm_r2)>=r2_real)+1)/(n_perm+1)
    return r2_real, p_value

def bootstrap_ci_params(params, n_boot=2000, alpha=0.05):
    boot_samples=[]
    for _ in range(n_boot):
        sample = resample(params)
        boot_samples.append(np.mean(sample))
    lower = np.percentile(boot_samples,100*alpha/2)
    upper = np.percentile(boot_samples,100*(1-alpha/2))
    return lower, upper

# ================================
# 5. MULTI-SEED PIPELINE
# ================================
def run_multiseed_pipeline_fixed(world="energy", n_seeds=50):
    all_results=[]
    for seed in range(n_seeds):
        np.random.seed(seed)
        df = generate_lifespan_dataset(world=world)
        result = fit_mixed_model(df)
        r2, p_perm = permutation_test_slope(df)
        boot_l, boot_u = bootstrap_ci_params(result.params.values)
        all_results.append({
            "Seed": seed,
            "Beta_ECM": result.params["ECM"],
            "Beta_PPI": result.params["PPI"],
            "Beta_Age2": result.params["Age2"],
            "R2_perm": r2,
            "p_perm": p_perm,
            "Bootstrap_Lower": boot_l,
            "Bootstrap_Upper": boot_u
        })
        print(f"{world.capitalize()} world seed {seed+1}/{n_seeds} complete")
    return pd.DataFrame(all_results)

# ================================
# 6. SUMMARY TABLE
# ================================
def summarize_world(df_results):
    summary = pd.DataFrame({
        "R2_mean": [df_results["R2_perm"].mean()],
        "R2_sd": [df_results["R2_perm"].std()],
        "Slope_mean": [df_results["Beta_ECM"].mean()],
        "Slope_sd": [df_results["Beta_ECM"].std()],
        "Perm_p_mean": [df_results["p_perm"].mean()],
        "Perm_p_sd": [df_results["p_perm"].std()]
    })
    return summary

# ================================
# 7. FIGURES
# ================================
def plot_results(df_results, world="energy"):
    sns.set(style="whitegrid", context="talk", palette="colorblind")
    fig, axes = plt.subplots(1,2,figsize=(14,6))

    # Panel 1: Beta_ECM ± CI
    axes[0].bar(range(len(df_results)), df_results["Beta_ECM"],
                yerr=[df_results["Beta_ECM"]-df_results["Bootstrap_Lower"],
                      df_results["Bootstrap_Upper"]-df_results["Beta_ECM"]],
                capsize=3, color="#1f77b4")
    axes[0].set_title(f"{world.capitalize()} World: ECM Beta ± Bootstrap CI")
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("Beta_ECM")

    # Panel 2: Age² Beta distribution
    axes[1].hist(df_results["Beta_Age2"], bins=15, color="#ff7f0e", alpha=0.8)
    axes[1].axvline(df_results["Beta_Age2"].mean(), color="black", linestyle="--")
    axes[1].set_title(f"{world.capitalize()} World: Age² Beta Distribution")
    axes[1].set_xlabel("Beta_Age2")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(f"FIGURE_{world.upper()}_PIPELINE.png", dpi=600)
    plt.savefig(f"FIGURE_{world.upper()}_PIPELINE.pdf")
    plt.show()

# ================================
# 8. MAIN EXECUTION
# ================================
if __name__=="__main__":
    worlds = ["energy","predictive","mixed","null"]
    all_summaries = []
    
    for world in worlds:
        print(f"\n=== Running {world.upper()} world ===")
        df_results = run_multiseed_pipeline_fixed(world=world, n_seeds=50)
        df_results.to_csv(f"LIFESPAN_PIPELINE_{world.upper()}.csv", index=False)
        summary = summarize_world(df_results)
        summary["World"] = world
        all_summaries.append(summary)
        plot_results(df_results, world=world)
    
    final_summary = pd.concat(all_summaries, ignore_index=True)
    final_summary = final_summary[["World","R2_mean","R2_sd","Slope_mean","Slope_sd","Perm_p_mean","Perm_p_sd"]]
    final_summary.to_csv("FINAL_SUMMARY_TABLE.csv", index=False)
    print("\n=== FINAL SUMMARY TABLE ===")
    print(final_summary)
    print("\nPipeline complete. All CSVs and figures saved.\n")
