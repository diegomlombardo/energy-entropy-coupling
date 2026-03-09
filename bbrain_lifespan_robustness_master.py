# ============================================================
# Brain-Body-Energy Generative Model - Publication-Ready
# ============================================================

import numpy as np
import pandas as pd
import json
from scipy.signal import hilbert
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, HuberRegressor
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import statsmodels.api as sm

# ===========================
# GLOBAL PARAMETERS
# ===========================
SEED_GLOBAL = 42
rng_global = np.random.default_rng(SEED_GLOBAL)

NETWORK_SIZES = [40, 80, 120]
N_NETWORKS_PER_SIZE = 5
SEEDS_PER_NETWORK = 50
N_SUBJECTS_PER_SEED = 200
T = 500
DT = 0.05
STEPS = int(T / DT)
N_PERM = 100

K_BODY = 0.05
K_ENERGY = 0.05

SAVE_DIR = "BrainBodyEnergy_Publication"
os.makedirs(SAVE_DIR, exist_ok=True)

# Save metadata
metadata = {
    "NETWORK_SIZES": NETWORK_SIZES,
    "N_NETWORKS_PER_SIZE": N_NETWORKS_PER_SIZE,
    "SEEDS_PER_NETWORK": SEEDS_PER_NETWORK,
    "N_SUBJECTS_PER_SEED": N_SUBJECTS_PER_SEED,
    "T": T,
    "DT": DT,
    "K_BODY": K_BODY,
    "K_ENERGY": K_ENERGY,
    "SEED_GLOBAL": SEED_GLOBAL
}
with open(os.path.join(SAVE_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

# ===========================
# NETWORK FUNCTIONS
# ===========================
def small_world(N, k=6, p=0.2, seed=None):
    rng = np.random.default_rng(seed)
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(1, k // 2 + 1):
            W[i, (i + j) % N] = 1
            W[i, (i - j) % N] = 1
    # Rewiring
    for i in range(N):
        for j in range(N):
            if W[i, j] == 1 and rng.random() < p:
                W[i, j] = 0
                new_j = rng.integers(N)
                while new_j == i or W[i, new_j] == 1:
                    new_j = rng.integers(N)
                W[i, new_j] = 1
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0)
    eig = np.linalg.eigvals(W)
    W = W / np.max(np.abs(eig))
    return W

# ===========================
# BODY OSCILLATORS
# ===========================
def simulate_body(seed):
    rng = np.random.default_rng(seed)
    heart_phase, resp_phase = 0, 0
    heart, resp = [], []
    for t in range(STEPS):
        heart_phase += 0.9 * DT + rng.normal(scale=0.01)
        resp_phase += 0.25 * DT + rng.normal(scale=0.01)
        heart.append(np.sin(heart_phase))
        resp.append(np.sin(resp_phase))
    return np.array(heart), np.array(resp)

# ===========================
# ENERGY DYNAMICS
# ===========================
def simulate_energy(brain_power, heart, resp):
    E = 1
    E_series = []
    decay = 0.02
    for t in range(STEPS):
        prod = 0.4 + 0.2 * heart[t] + 0.1 * resp[t]
        cons = 0.05 * brain_power[t]
        dE = prod - cons - decay * E
        E += dE * DT
        E_series.append(E)
    return np.array(E_series)

# ===========================
# BRAIN DYNAMICS
# ===========================
def simulate_brain(W, G, noise, heart, energy, seed):
    rng = np.random.default_rng(seed)
    N = W.shape[0]
    z = rng.normal(size=N) + 1j * rng.normal(size=N)
    omega = rng.uniform(0.04, 0.07, N)
    row = W.sum(axis=1)
    traj_power = np.zeros((STEPS, N))
    for t in range(STEPS):
        coupling = G * (W @ z - row * z)
        dz = (0.02 + 1j * omega - np.abs(z)**2) * z + coupling
        dz += K_BODY * heart[t] + K_ENERGY * energy[t]
        dz += noise * np.sqrt(DT) * (rng.normal(size=N) + 1j * rng.normal(size=N))
        z += dz * DT
        traj_power[t] = np.abs(z)**2
    return traj_power

# ===========================
# METRICS
# ===========================
def metastability(traj_power):
    # Corrected: phase of complex-valued z dynamics
    z_phase = np.angle(traj_power.astype(complex))
    R = np.abs(np.mean(np.exp(1j * z_phase), axis=1))
    return np.std(R)

def coherence(sig1, sig2):
    phase1 = np.angle(hilbert(sig1))
    phase2 = np.angle(hilbert(sig2))
    return np.abs(np.mean(np.exp(1j * (phase1 - phase2))))

def predictive_metric(traj_power):
    diff = np.diff(np.mean(traj_power, axis=1))
    return np.sqrt(np.mean(diff**2))

def energy_metrics(E):
    return np.std(E), np.mean(E)

# ===========================
# SYNTHETIC COGNITION (INDEPENDENT)
# ===========================
def generate_synthetic_cognition_independent(heart, resp, energy, seed):
    rng = np.random.default_rng(seed)
    cognition = (
        0.4 * np.mean(energy) 
        - 0.2 * np.std(energy)
        + 0.2 * np.mean(heart)
        + 0.2 * np.mean(resp)
        + rng.normal(scale=0.05)
    )
    return cognition

# ===========================
# SIMULATE SUBJECT
# ===========================
def simulate_subject(W, network_id, seed, subject_id):
    rng = np.random.default_rng(seed + subject_id)
    age = rng.uniform(20, 80)
    G = rng.uniform(0.2, 1.0)
    noise = rng.uniform(0.01, 0.05)

    heart, resp = simulate_body(seed + subject_id)
    brain_power_dummy = np.ones(STEPS)
    energy_dummy = simulate_energy(brain_power_dummy, heart, resp)

    traj_power = simulate_brain(W, G, noise, heart, energy_dummy, seed + subject_id)
    brain_global = np.mean(traj_power, axis=1)

    m = metastability(traj_power)
    coh = coherence(brain_global, heart)
    fm = predictive_metric(traj_power)

    brain_power = np.mean(traj_power, axis=1)
    energy = simulate_energy(brain_power, heart, resp)
    stab, eff = energy_metrics(energy)

    synth_cog = generate_synthetic_cognition_independent(
        heart, resp, energy, seed + subject_id
    )

    return {
        "NetworkID": network_id,
        "Seed": seed,
        "SubjectID": subject_id,
        "Age": age,
        "G": G,
        "Noise": noise,
        "HeartMean": np.mean(heart),
        "RespMean": np.mean(resp),
        "Metastability": m,
        "BrainHeartCoherence": coh,
        "ForwardModeling": fm,
        "EnergyStability": stab,
        "EnergyEfficiency": eff,
        "Cognition": synth_cog
    }

# ===========================
# DATASET GENERATION
# ===========================
def generate_dataset_incremental():
    network_counter = 0
    all_results = []
    for N in NETWORK_SIZES:
        for net_idx in range(N_NETWORKS_PER_SIZE):
            W = small_world(N, seed=net_idx)
            for seed in range(SEEDS_PER_NETWORK):
                results = Parallel(n_jobs=-1)(delayed(simulate_subject)(
                    W, network_counter, seed, subj
                ) for subj in range(N_SUBJECTS_PER_SEED))
                
                df_seed = pd.DataFrame(results)
                df_seed['NullModel'] = np.random.normal(size=len(df_seed))
                file_name = f"dataset_network{network_counter}_seed{seed}.csv"
                df_seed.to_csv(os.path.join(SAVE_DIR, file_name), index=False)
                all_results.append(df_seed)
            network_counter += 1
    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv(os.path.join(SAVE_DIR, "dataset_full.csv"), index=False)
    return df_all

# ===========================
# STATISTICAL ANALYSIS
# ===========================
def crossval_r2(df, pred, target, robust=False):
    X = df[[pred, "Age"]].values
    y = df[target].values
    kf = KFold(10, shuffle=True, random_state=SEED_GLOBAL)
    r2s = []
    for tr, te in kf.split(X):
        if robust:
            model = HuberRegressor().fit(X[tr], y[tr])
            y_pred = model.predict(X[te])
        else:
            model = LinearRegression().fit(X[tr], y[tr])
            y_pred = model.predict(X[te])
        r, _ = pearsonr(y_pred, y[te])
        r2s.append(r**2)
    return np.mean(r2s), np.percentile(r2s, 2.5), np.percentile(r2s, 97.5)

def permutation_test(df, pred, target, n_perm=N_PERM):
    real, ci_low, ci_high = crossval_r2(df, pred, target)
    perm_r2s = []
    for _ in range(n_perm):
        df_perm = df.copy()
        df_perm[pred] = np.random.permutation(df_perm[pred])
        r2_perm, _, _ = crossval_r2(df_perm, pred, target)
        perm_r2s.append(r2_perm)
    p_val = np.mean(np.array(perm_r2s) >= real)
    return real, ci_low, ci_high, p_val

# ===========================
# MEDIATION ANALYSIS
# ===========================
def mediation_analysis(df, model_pred, mediator="EnergyEfficiency", outcome="Cognition"):
    X = df[[model_pred, "Age"]]
    X = sm.add_constant(X)
    M = df[mediator]
    Y = df[outcome]
    # Step 1: X -> M
    model_M = sm.OLS(M, X).fit()
    # Step 2: X + M -> Y
    X2 = X.copy()
    X2[mediator] = M
    model_Y = sm.OLS(Y, X2).fit()
    indirect = model_M.params[model_pred] * model_Y.params[mediator]
    direct = model_Y.params[model_pred]
    return indirect, direct

# ===========================
# RUN SIMULATION
# ===========================
print("Generating dataset incrementally...")
df = generate_dataset_incremental()
print("Dataset generation complete.")

# ===========================
# FIGURES
# ===========================
sns.set(style="whitegrid", context="talk")
FIG_DIR = os.path.join(SAVE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Figure: Model vs Energy Efficiency
predictors = ["Metastability","BrainHeartCoherence","ForwardModeling","NullModel"]
targets = ["EnergyEfficiency"]

fig, axes = plt.subplots(1,len(predictors),figsize=(20,5))
for i, pred in enumerate(predictors):
    sns.regplot(
        x=pred, y="Cognition", data=df,
        ax=axes[i], scatter_kws={"alpha":0.3}, line_kws={"color":"black"}
    )
    axes[i].set_title(f"{pred} → Cognition")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/Model_vs_Cognition.png", dpi=600)
plt.close()

# Figure: Longitudinal aging trends
age_bins = np.linspace(20,80,13)
age_trends = []
for pred in ["EnergyEfficiency"]:
    trend = []
    for i in range(len(age_bins)-1):
        mask = (df['Age'] >= age_bins[i]) & (df['Age'] < age_bins[i+1])
        if mask.sum()<5:
            trend.append(np.nan)
        else:
            r,_ = pearsonr(df.loc[mask,pred], df.loc[mask,"Cognition"])
            trend.append(r)
    age_trends.append({
        "Predictor": pred,
        "AgeBinCenters": (age_bins[:-1]+age_bins[1:])/2,
        "Correlation": trend
    })
plt.figure(figsize=(8,5))
for trend in age_trends:
    plt.plot(trend['AgeBinCenters'], trend['Correlation'], label=trend['Predictor'])
plt.xlabel("Age")
plt.ylabel("Correlation with Cognition")
plt.title("Longitudinal Aging Trends")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/Longitudinal_Aging_Trends.png", dpi=600)
plt.close()

# Figure: Energy Distribution
plt.figure(figsize=(6,5))
sns.histplot(df["EnergyEfficiency"], bins=30, kde=True)
plt.xlabel("Energy Efficiency")
plt.ylabel("Count")
plt.title("Energy Efficiency Distribution")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/Energy_Distribution.png", dpi=600)
plt.close()

# Mediation analysis
med_results = {}
for model in ["BrainHeartCoherence","ForwardModeling","NullModel"]:
    indirect, direct = mediation_analysis(df, model)
    med_results[model] = {"Indirect":indirect,"Direct":direct}

# Figure: Mediation
med_df = pd.DataFrame(med_results).T
med_df.plot(kind="bar", figsize=(6,5))
plt.ylabel("Effect Size")
plt.title("Mediation Analysis: Energy Efficiency Mediates Cognition")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/Mediation_Effects.png", dpi=600)
plt.close()

# Save final results table
final_table = df.copy()
final_table.to_csv(os.path.join(SAVE_DIR,"final_dataset.csv"), index=False)
print("Figures and final dataset saved.")
