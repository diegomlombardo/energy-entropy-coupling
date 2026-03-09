# ============================================================
# Brain-Body-Energy Generative Model - Robust & Non-Circular Cognition
# ============================================================

import numpy as np
import pandas as pd
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
from statsmodels.stats.mediation import Mediation

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

# ===========================
# NETWORK GENERATION
# ===========================
def small_world(N, k=6, p=0.2, seed=None):
    rng = np.random.default_rng(seed)
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(1, k//2 + 1):
            W[i, (i+j) % N] = 1
            W[i, (i-j) % N] = 1
    # Rewiring
    for i in range(N):
        for j in range(N):
            if W[i,j] == 1 and rng.random() < p:
                W[i,j] = 0
                new_j = rng.integers(N)
                while new_j == i or W[i,new_j] == 1:
                    new_j = rng.integers(N)
                W[i,new_j] = 1
    W = (W + W.T)/2
    np.fill_diagonal(W,0)
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
        prod = 0.4 + 0.2*heart[t] + 0.1*resp[t]
        cons = 0.05 * brain_power[t]
        dE = prod - cons - decay*E
        E += dE*DT
        E_series.append(E)
    return np.array(E_series)

# ===========================
# BRAIN DYNAMICS
# ===========================
def simulate_brain(W, G, noise, heart, energy, seed):
    rng = np.random.default_rng(seed)
    N = W.shape[0]
    z = rng.normal(size=N) + 1j*rng.normal(size=N)
    omega = rng.uniform(0.04, 0.07, N)
    row = W.sum(axis=1)
    traj_power = np.zeros((STEPS, N))
    for t in range(STEPS):
        coupling = G*(W @ z - row*z)
        dz = (0.02 + 1j*omega - np.abs(z)**2)*z + coupling
        dz += K_BODY*heart[t] + K_ENERGY*energy[t]
        dz += noise*np.sqrt(DT)*(rng.normal(size=N)+1j*rng.normal(size=N))
        z += dz*DT
        traj_power[t] = np.abs(z)**2
    return traj_power

# ===========================
# METRICS
# ===========================
def metastability(traj_power):
    phase = np.angle(traj_power.astype(complex))
    R = np.abs(np.mean(np.exp(1j*phase), axis=1))
    return np.std(R)

def coherence(sig1, sig2):
    phase1 = np.angle(hilbert(sig1))
    phase2 = np.angle(hilbert(sig2))
    return np.abs(np.mean(np.exp(1j*(phase1-phase2))))

def predictive_metric(traj_power):
    amp = traj_power
    diff = np.diff(np.mean(amp, axis=1))
    return np.sqrt(np.mean(diff**2))

def energy_metrics(E):
    return np.std(E), np.mean(E)

# ===========================
# NON-CIRCULAR SYNTHETIC COGNITION
# ===========================
def generate_cognition(heart, resp, energy):
    """
    Cognition independent of brain metrics.
    Scientifically reasonable: function of body signals and energy stats.
    """
    energy_mean = np.mean(energy)
    energy_std = np.std(energy)
    heart_mean = np.mean(heart)
    resp_mean = np.mean(resp)
    rng = np.random.default_rng()
    cognition = 0.4*energy_mean - 0.2*energy_std + 0.2*heart_mean + 0.2*resp_mean + rng.normal(scale=0.05)
    return cognition

# ===========================
# SIMULATE SUBJECT
# ===========================
def simulate_subject(W, network_id, seed, subject_id):
    rng = np.random.default_rng(seed+subject_id)
    age = rng.uniform(20,80)
    G = rng.uniform(0.2,1.0)
    noise = rng.uniform(0.01,0.05)

    heart, resp = simulate_body(seed+subject_id)
    brain_power_dummy = np.ones(STEPS)
    energy_dummy = simulate_energy(brain_power_dummy, heart, resp)

    traj_power = simulate_brain(W, G, noise, heart, energy_dummy, seed+subject_id)
    brain_global = np.mean(traj_power, axis=1)

    m = metastability(traj_power)
    coh = coherence(brain_global, heart)
    fm = predictive_metric(traj_power)

    brain_power = np.mean(traj_power, axis=1)
    energy = simulate_energy(brain_power, heart, resp)
    stab, eff = energy_metrics(energy)

    cognition = generate_cognition(heart, resp, energy)

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
        "SyntheticCognition": cognition
    }

# ===========================
# GENERATE DATASET
# ===========================
def generate_dataset():
    all_results = []
    network_counter = 0
    for N in NETWORK_SIZES:
        for net_idx in range(N_NETWORKS_PER_SIZE):
            W = small_world(N, seed=net_idx)
            for seed in range(SEEDS_PER_NETWORK):
                results = Parallel(n_jobs=-1)(delayed(simulate_subject)(
                    W, network_counter, seed, subj
                ) for subj in range(N_SUBJECTS_PER_SEED))
                df_seed = pd.DataFrame(results)
                df_seed['NullModel'] = np.random.normal(size=len(df_seed))
                all_results.append(df_seed)
            network_counter +=1
    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv(os.path.join(SAVE_DIR, "dataset_full.csv"), index=False)
    return df_all

# ===========================
# CROSS-VALIDATION & PERMUTATION
# ===========================
def crossval_r2(df, pred, target, robust=False):
    X = df[[pred,'Age']].values
    y = df[target].values
    kf = KFold(10, shuffle=True, random_state=SEED_GLOBAL)
    r2s = []
    for tr, te in kf.split(X):
        if robust:
            model = HuberRegressor().fit(X[tr], y[tr])
        else:
            model = LinearRegression().fit(X[tr], y[tr])
        y_pred = model.predict(X[te])
        r, _ = pearsonr(y_pred, y[te])
        r2s.append(r**2)
    return np.mean(r2s), np.percentile(r2s,2.5), np.percentile(r2s,97.5)

def permutation_test(df, pred, target, n_perm=100):
    real, _, _ = crossval_r2(df, pred, target)
    perm_r2s = []
    for _ in range(n_perm):
        df_perm = df.copy()
        df_perm[pred] = np.random.permutation(df_perm[pred])
        r2_perm, _, _ = crossval_r2(df_perm, pred, target)
        perm_r2s.append(r2_perm)
    p_val = np.mean(np.array(perm_r2s) >= real)
    return real, p_val

# ===========================
# MEDIATION ANALYSIS
# ===========================
def run_mediation(df, predictor, mediator, outcome):
    # Predictor -> Mediator -> Outcome
    X = sm.add_constant(df[[predictor,'Age']])
    M = df[mediator]
    Y = df[outcome]
    # Step 1: effect of predictor on mediator
    model_m = sm.OLS(M, X).fit()
    # Step 2: effect of predictor and mediator on outcome
    X2 = sm.add_constant(df[[predictor, mediator, 'Age']])
    model_y = sm.OLS(Y, X2).fit()
    med = Mediation(model_y, model_m, mediator_name=mediator, exposure_name=predictor)
    med_res = med.fit(n_rep=100)
    return med_res

# ===========================
# MAIN: GENERATE DATA & ANALYSES
# ===========================
print("Generating dataset...")
df = generate_dataset()
print("Dataset generated with", len(df), "subjects")

# -----------------------------
# Predictive models
# -----------------------------
predictors = ["Metastability", "BrainHeartCoherence", "ForwardModeling", "NullModel"]
target = "SyntheticCognition"

results = []
for pred in predictors:
    r2_mean, r2_low, r2_high = crossval_r2(df, pred, target)
    real_r2, p_val = permutation_test(df, pred, target)
    results.append({
        "Predictor": pred,
        "Target": target,
        "CV_R2": r2_mean,
        "CI_lower": r2_low,
        "CI_upper": r2_high,
        "Permutation_p": p_val
    })

res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(SAVE_DIR,"predictive_models_cognition.csv"), index=False)
print(res_df)

# -----------------------------
# Mediation analysis: Models -> EnergyEfficiency -> Cognition
# -----------------------------
med_results = []
for pred in predictors:
    med_res = run_mediation(df, predictor=pred, mediator="EnergyEfficiency", outcome="SyntheticCognition")
    med_results.append({
        "Predictor": pred,
        "ACME": med_res.acme_avg,
        "ADE": med_res.ade_avg,
        "TotalEffect": med_res.total_effect,
        "PropMediated": med_res.prop_med
    })

med_df = pd.DataFrame(med_results)
med_df.to_csv(os.path.join(SAVE_DIR,"mediation_results.csv"), index=False)
print(med_df)

# -----------------------------
# FIGURES: Publication-level
# -----------------------------
sns.set(style="whitegrid")

# Cognition distribution
plt.figure(figsize=(7,5))
sns.histplot(df['SyntheticCognition'], kde=True, bins=50, color='skyblue')
plt.title("Synthetic Cognition Distribution")
plt.xlabel("Synthetic Cognition")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR,"fig_cognition_distribution.png"), dpi=300)
plt.show()

# Scatter: EnergyEfficiency vs Cognition
plt.figure(figsize=(7,5))
sns.scatterplot(x='EnergyEfficiency', y='SyntheticCognition', data=df, alpha=0.4)
sns.regplot(x='EnergyEfficiency', y='SyntheticCognition', data=df, scatter=False, color='red')
plt.title("Energy Efficiency vs Cognition")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR,"fig_energy_cognition.png"), dpi=300)
plt.show()

# Correlation heatmap
metrics = ["Metastability","BrainHeartCoherence","ForwardModeling","EnergyStability","EnergyEfficiency","SyntheticCognition"]
plt.figure(figsize=(8,6))
sns.heatmap(df[metrics].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Brain-Body-Energy Metrics Correlation")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR,"fig_correlation_heatmap.png"), dpi=300)
plt.show()

# Mediation plot
plt.figure(figsize=(7,5))
for i,row in med_df.iterrows():
    plt.bar(row['Predictor'], row['TotalEffect'], color='skyblue', alpha=0.6, label='TotalEffect')
    plt.bar(row['Predictor'], row['ACME'], color='orange', alpha=0.6, label='IndirectEffect')
plt.ylabel("Effect Size")
plt.title("Mediation: Models -> EnergyEfficiency -> Cognition")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR,"fig_mediation.png"), dpi=300)
plt.show()
