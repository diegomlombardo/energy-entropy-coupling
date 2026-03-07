# ============================================================
# Brain-Body-Energy Generative Model with Synthetic Cognition
# - Incremental saving
# - Cross-validated R²
# - Permutation tests
# - Longitudinal trends
# - Forward modeling predictor
# - Synthetic cognition integration
# ============================================================

import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

# ============================================================
# PARAMETERS
# ============================================================
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

# ============================================================
# NETWORK GENERATION
# ============================================================
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

# ============================================================
# BODY OSCILLATORS
# ============================================================
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

# ============================================================
# ENERGY SIMULATION
# ============================================================
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

# ============================================================
# BRAIN DYNAMICS
# ============================================================
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

# ============================================================
# METRICS
# ============================================================
def metastability(traj_power):
    phase = np.angle(traj_power.astype(complex))
    R = np.abs(np.mean(np.exp(1j * phase), axis=1))
    return np.std(R)

def coherence(sig1, sig2):
    phase1 = np.angle(hilbert(sig1))
    phase2 = np.angle(hilbert(sig2))
    return np.abs(np.mean(np.exp(1j * (phase1 - phase2))))

def predictive_metric(traj_power):
    amp = traj_power
    diff = np.diff(np.mean(amp, axis=1))
    return np.sqrt(np.mean(diff**2))

def energy_metrics(E):
    return np.std(E), np.mean(E)

# ============================================================
# SYNTHETIC COGNITION
# ============================================================
def generate_synthetic_cognition(traj_power, heart, resp, seed):
    rng = np.random.default_rng(seed)
    brain_global = np.mean(traj_power, axis=1)
    coh = coherence(brain_global, heart)
    fm = predictive_metric(traj_power)
    m = metastability(traj_power)
    # Weighted sum + noise
    cognition = 0.4 * fm + 0.3 * m + 0.3 * coh + rng.normal(scale=0.05)
    return cognition

# ============================================================
# SIMULATE SUBJECT
# ============================================================
def simulate_subject(W, network_id, seed, subject_id):
    rng = np.random.default_rng(seed + subject_id)
    age = rng.uniform(20, 80)
    G = rng.uniform(0.2, 1.0)
    noise = rng.uniform(0.01, 0.05)
    
    heart, resp = simulate_body(seed + subject_id)
    
    # dummy energy for initial brain simulation
    brain_power_dummy = np.ones(STEPS)
    energy_dummy = simulate_energy(brain_power_dummy, heart, resp)
    
    traj_power = simulate_brain(W, G, noise, heart, energy_dummy, seed + subject_id)
    brain_global = np.mean(traj_power, axis=1)
    
    m = metastability(traj_power)
    coh = coherence(brain_global, heart)
    fm = predictive_metric(traj_power)  # Forward Modeling
    
    brain_power = np.mean(traj_power, axis=1)
    energy = simulate_energy(brain_power, heart, resp)
    stab, eff = energy_metrics(energy)
    
    synth_cog = generate_synthetic_cognition(traj_power, heart, resp, seed + subject_id)
    
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
        "SyntheticCognition": synth_cog
    }

# ============================================================
# DATASET GENERATION WITH INCREMENTAL SAVING
# ============================================================
def generate_dataset_incremental():
    network_counter = 0
    all_results = []
    
    for N in NETWORK_SIZES:
        for net_idx in range(N_NETWORKS_PER_SIZE):
            print(f"Generating network {network_counter+1}/{len(NETWORK_SIZES)*N_NETWORKS_PER_SIZE} size {N}")
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

# ============================================================
# STATISTICAL ANALYSIS
# ============================================================
def crossval_r2(df, pred, target):
    X = df[[pred, "Age"]].values
    y = df[target].values
    kf = KFold(10, shuffle=True, random_state=42)
    r2s = []
    for tr, te in kf.split(X):
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

# ============================================================
# RUN FULL SIMULATION AND ANALYSIS
# ============================================================
print("Generating dataset incrementally...")
df = generate_dataset_incremental()
print("Dataset generation complete.")

# -----------------------------
# Model comparisons
# -----------------------------
predictors = ["Metastability", "BrainHeartCoherence", "ForwardModeling", "NullModel"]
targets = ["EnergyStability", "EnergyEfficiency"]

results = []
for pred in predictors:
    for target in targets:
        print(f"Analyzing {pred} → {target}")
        r2, ci_low, ci_high, pval = permutation_test(df, pred, target)
        results.append({
            "Predictor": pred,
            "Target": target,
            "CrossVal_R2": r2,
            "CI_lower": ci_low,
            "CI_upper": ci_high,
            "p_value": pval
        })

res_df = pd.DataFrame(results)
res_df["p_adj_bonf"] = multipletests(res_df["p_value"], method='bonferroni')[1]
res_df["p_adj_fdr"] = multipletests(res_df["p_value"], method='fdr_bh')[1]
res_df.to_excel(os.path.join(SAVE_DIR, "model_comparison.xlsx"), index=False)
print("Energy model analysis complete.")
print(res_df)

# -----------------------------
# Synthetic Cognition Analysis
# -----------------------------
cog_predictors = ["Metastability", "BrainHeartCoherence", "ForwardModeling"]
cog_target = "SyntheticCognition"

cog_results = []
for pred in cog_predictors:
    r2, ci_low, ci_high, pval = permutation_test(df, pred, cog_target)
    cog_results.append({
        "Predictor": pred,
        "Target": cog_target,
        "CrossVal_R2": r2,
        "CI_lower": ci_low,
        "CI_upper": ci_high,
        "p_value": pval
    })

cog_df = pd.DataFrame(cog_results)
cog_df["p_adj_bonf"] = multipletests(cog_df["p_value"], method='bonferroni')[1]
cog_df["p_adj_fdr"] = multipletests(cog_df["p_value"], method='fdr_bh')[1]
cog_df.to_excel(os.path.join(SAVE_DIR, "synthetic_cognition_analysis.xlsx"), index=False)
print("Synthetic cognition analysis complete.")
print(cog_df)

# ============================================================
# PUBLICATION-QUALITY FIGURES
# ============================================================
sns.set(style="whitegrid")

# Model comparison barplot
plt.figure(figsize=(8,5))
sns.barplot(data=res_df, x="Predictor", y="CrossVal_R2", hue="Target", capsize=0.2)
plt.title("Predictors of Energy Stability and Efficiency")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "model_comparison_barplot.png"), dpi=300)
plt.show()

# Synthetic cognition barplot
plt.figure(figsize=(6,4))
sns.barplot(data=cog_df, x="Predictor", y="CrossVal_R2", capsize=0.2, color="skyblue")
plt.title("Predictors of Synthetic Cognition")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "synthetic_cognition_barplot.png"), dpi=300)
plt.show()

# Predictor correlation heatmap including Synthetic Cognition
plt.figure(figsize=(7,6))
corr = df[["Metastability","BrainHeartCoherence","ForwardModeling","SyntheticCognition"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Predictor Correlation Including Synthetic Cognition")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "predictor_correlation_synthcog.png"), dpi=300)
plt.show()

# Age-dependent trends for energy metrics
age_bins = np.linspace(20, 80, 13)
age_trends = []
for pred in predictors:
    for target in targets:
        trend = []
        for i in range(len(age_bins)-1):
            mask = (df['Age'] >= age_bins[i]) & (df['Age'] < age_bins[i+1])
            if mask.sum() < 5:
                trend.append(np.nan)
            else:
                r, _ = pearsonr(df.loc[mask, pred], df.loc[mask, target])
                trend.append(r)
        age_trends.append({
            "Predictor": pred,
            "Target": target,
            "AgeBinCenters": (age_bins[:-1]+age_bins[1:])/2,
            "Correlation": trend
        })

with open(os.path.join(SAVE_DIR, "longitudinal_trends.pkl"), "wb") as f:
    pickle.dump(age_trends, f)

plt.figure(figsize=(10,6))
for trend in age_trends:
    plt.plot(trend['AgeBinCenters'], trend['Correlation'], label=f"{trend['Predictor']} → {trend['Target']}")
plt.xlabel("Age")
plt.ylabel("Correlation with Energy Metrics")
plt.title("Age-dependent Predictor Vulnerability")
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "longitudinal_trends.png"), dpi=300)
plt.show()

# Peripheral physiology vs energy efficiency
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="HeartMean", y="EnergyEfficiency", alpha=0.3, label="Heart")
sns.scatterplot(data=df, x="RespMean", y="EnergyEfficiency", alpha=0.3, label="Respiration")
plt.title("Peripheral Physiology vs Energy Efficiency")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "peripheral_vs_energy.png"), dpi=300)
plt.show()
