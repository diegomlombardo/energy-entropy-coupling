#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Brain–Body Lifespan Simulation Pipeline – Biologically Plausible Q1 Version
Features
--------
• Modular neural network (small-world connectivity)
• Age-dependent decline in neural and body dynamics
• Physiologically inspired cardiac, respiratory, metabolic oscillators
• ECM and PPI mapped to cognitive domains
• Multi-seed robustness and permutation testing
• Mixed-effects longitudinal regression with random slopes
• Parallelized execution and numba acceleration
• Publication-quality figures
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
import networkx as nx
from numba import njit

warnings.filterwarnings("ignore")

# =====================================================
# RANDOM GENERATOR
# =====================================================
def get_rng(seed):
    return np.random.default_rng(seed)

# =====================================================
# BRAIN–BODY DYNAMICS
# =====================================================
def create_small_world_adj(N=40, k=4, p=0.1, seed=None):
    """Return small-world adjacency matrix for N nodes."""
    G = nx.watts_strogatz_graph(N, k, p, seed=int(seed) if seed is not None else None)
    return nx.to_numpy_array(G)

@njit
def body_update(cardiac, resp, metabolic, R, dt, age_factor):
    """Physiologically inspired body update per timestep."""
    d_cardiac = 0.3*cardiac - cardiac**3 + 0.04*R
    d_resp = 0.25*resp - resp**3 + 0.03*R
    d_met = 0.1*metabolic - metabolic**3 + 0.02*R
    # Apply age-related decline
    d_cardiac *= age_factor
    d_resp *= age_factor
    d_met *= age_factor
    return cardiac + d_cardiac*dt, resp + d_resp*dt, metabolic + d_met*dt

def simulate_system(rng, N=40, T=300, dt=0.02, coupling=0.2, body_feedback=0.05, noise=0.02, age_factor=1.0):
    """Simulate biologically plausible brain-body system with structured network."""
    omega = rng.uniform(0.04,0.07,N)
    z = rng.normal(size=N) + 1j*rng.normal(size=N)
    cardiac = respiratory = metabolic = 0.1
    steps = int(T/dt)
    Z = np.zeros((steps,N),dtype=complex)
    P = np.zeros((steps,3))
    adj = create_small_world_adj(N=N, k=4, p=0.1, seed=int(rng.integers(0,1e6)))

    for t in range(steps):
        phases = np.angle(z)
        R = np.abs(np.exp(1j*phases).mean())
        cardiac, respiratory, metabolic = body_update(cardiac, respiratory, metabolic, R, dt, age_factor)
        body_state = (cardiac + respiratory + metabolic)/3

        dz = (0.02 + 1j*omega - np.abs(z)**2)*z
        # Coupling through adjacency
        dz += coupling*(adj @ z - np.sum(adj,axis=1)*z)
        dz += body_feedback*body_state*z
        dz += noise*(rng.normal(size=N)+1j*rng.normal(size=N))
        z += dz*dt

        Z[t] = z
        P[t] = [cardiac, respiratory, metabolic]

    return Z, P

# =====================================================
# METRICS
# =====================================================
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

def compute_ECM(Z,P):
    _,R = metastability(Z)
    H = entropy_signal(R)
    body_mean = np.mean(P,axis=1)
    MI = gaussian_copula_mi(R,body_mean)
    return MI - H

def compute_PPI(Z):
    X = np.abs(Z)
    X_pred = X[:-1]
    X_true = X[1:]
    beta = np.linalg.pinv(X_pred) @ X_true
    residual = X_true - X_pred @ beta
    return np.mean(residual**2)

# =====================================================
# DATASET GENERATION
# =====================================================
def generate_dataset(seed, world, subjects=60, timepoints=6):
    rng_main = get_rng(seed)
    ages = np.linspace(10,80,timepoints)
    rows=[]

    for subj in range(subjects):
        rng_subj = get_rng(seed + subj)
        subj_shift = rng_subj.normal(0,0.05)

        for age in ages:
            # Age factor for neural & body decline
            age_factor = 1 - 0.01*((age-45)/45)

            if world=="energy":
                Z,P = simulate_system(rng_subj, coupling=0.25*age_factor+subj_shift, body_feedback=0.08, age_factor=age_factor)
                latent = np.mean(np.abs(Z))   # energy-related
            elif world=="predictive":
                Z,P = simulate_system(rng_subj, coupling=0.15*age_factor+subj_shift, body_feedback=0.02, age_factor=age_factor)
                latent = np.var(np.diff(np.abs(Z),axis=0))  # predictive signal
            else: # null
                Z,P = simulate_system(rng_subj, age_factor=age_factor)
                latent = rng_subj.normal()

            ECM = compute_ECM(Z,P)
            PPI = compute_PPI(Z)
            # Map metrics to cognitive domains
            Cog_attention = 0.5*ECM + 0.3*latent + rng_subj.normal(0,0.3)
            Cog_memory = 0.6*PPI + 0.2*latent + rng_subj.normal(0,0.3)
            Cog = 0.5*(Cog_attention + Cog_memory)

            rows.append([subj, age, ECM, PPI, Cog])

    df = pd.DataFrame(rows, columns=["Subject","Age","ECM","PPI","Cog"])
    df["Age_c"] = df.Age - df.Age.mean()
    df["Age2"] = df.Age_c**2
    return df

# =====================================================
# MIXED MODEL
# =====================================================
def fit_mixed_model(df):
    model = smf.mixedlm("Cog ~ ECM + PPI + Age_c + Age2", df, groups=df["Subject"], re_formula="~ECM+PPI")
    return model.fit()

# =====================================================
# PERMUTATION TEST
# =====================================================
def permutation_test(df, n_perm=500):
    X = df[["ECM","PPI","Age_c","Age2"]].values
    y = df["Cog"].values
    real_model = LinearRegression().fit(X,y)
    r2_real = r2_score(y,real_model.predict(X))
    perm_r2 = []
    for _ in range(n_perm):
        y_perm = np.random.permutation(y)
        perm_r2.append(r2_score(y_perm, LinearRegression().fit(X,y_perm).predict(X)))
    p_value = (np.sum(np.array(perm_r2) >= r2_real)+1)/(n_perm+1)
    return r2_real, p_value

# =====================================================
# MULTI-SEED PIPELINE
# =====================================================
def run_pipeline_single_seed(seed, world):
    df = generate_dataset(seed, world)
    result = fit_mixed_model(df)
    r2, p_perm = permutation_test(df)
    return {
        "seed": seed,
        "beta_ECM": result.params.get("ECM",np.nan),
        "beta_PPI": result.params.get("PPI",np.nan),
        "beta_Age2": result.params.get("Age2",np.nan),
        "R2": r2,
        "perm_p": p_perm
    }

def run_pipeline(world, seeds=50):
    results = Parallel(n_jobs=-1)(delayed(run_pipeline_single_seed)(seed, world) for seed in range(seeds))
    return pd.DataFrame(results)

# =====================================================
# SUMMARY
# =====================================================
def summarize_results(df):
    return pd.DataFrame({
        "R2_mean":[df.R2.mean()],
        "R2_sd":[df.R2.std()],
        "Beta_ECM_mean":[df.beta_ECM.mean()],
        "Beta_PPI_mean":[df.beta_PPI.mean()],
        "Age2_mean":[df.beta_Age2.mean()],
        "Perm_p_mean":[df.perm_p.mean()]
    })

# =====================================================
# FIGURES
# =====================================================
def plot_results(df, world):
    sns.set(style="whitegrid",context="talk")
    fig, axes = plt.subplots(1,2,figsize=(14,6))
    sns.histplot(df.beta_ECM,kde=True,ax=axes[0])
    axes[0].set_title(f"{world} world – ECM effect")
    sns.histplot(df.beta_PPI,kde=True,ax=axes[1])
    axes[1].set_title(f"{world} world – PPI effect")
    plt.tight_layout()
    plt.savefig(f"FIG_{world}.png",dpi=600)
    plt.savefig(f"FIG_{world}.pdf")
    plt.close()

# =====================================================
# MAIN
# =====================================================
if __name__=="__main__":
    worlds = ["energy","predictive","null"]
    summaries=[]
    for world in worlds:
        print("\nRunning world:", world)
        results = run_pipeline(world,seeds=50)
        results.to_csv(f"RESULTS_{world}.csv",index=False)
        summary = summarize_results(results)
        summary["world"]=world
        summaries.append(summary)
        plot_results(results, world)
    final_summary = pd.concat(summaries, ignore_index=True)
    final_summary.to_csv("FINAL_SUMMARY_TABLE.csv",index=False)
    print("\nFINAL SUMMARY")
    print(final_summary)
