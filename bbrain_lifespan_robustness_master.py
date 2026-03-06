# ============================================================
# Brain-Body-Energy Generative Model
# ============================================================

import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

SAVE_DIR = "BrainBodyEnergy_Q1_Upgraded"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# NETWORK GENERATION
# ============================================================
def small_world(N, k=6, p=0.2, seed=None):
    rng = np.random.default_rng(seed)
    W = np.zeros((N,N))
    for i in range(N):
        for j in range(1,k//2+1):
            W[i,(i+j)%N] = 1
            W[i,(i-j)%N] = 1
    # Rewiring
    for i in range(N):
        for j in range(N):
            if W[i,j]==1 and rng.random()<p:
                W[i,j]=0
                new_j = rng.integers(N)
                while new_j==i or W[i,new_j]==1:
                    new_j = rng.integers(N)
                W[i,new_j]=1
    W = (W+W.T)/2
    np.fill_diagonal(W,0)
    eig = np.linalg.eigvals(W)
    W = W/np.max(np.abs(eig))
    return W

# ============================================================
# BODY OSCILLATORS
# ============================================================
def simulate_body(seed):
    rng = np.random.default_rng(seed)
    heart_phase, resp_phase = 0, 0
    heart, resp = [], []
    for t in range(STEPS):
        heart_phase += 0.9*DT + rng.normal(scale=0.01)
        resp_phase += 0.25*DT + rng.normal(scale=0.01)
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
        prod = 0.4 + 0.2*heart[t] + 0.1*resp[t]
        cons = 0.05*brain_power[t]
        dE = prod - cons - decay*E
        E += dE*DT
        E_series.append(E)
    return np.array(E_series)

# ============================================================
# BRAIN DYNAMICS
# ============================================================
def simulate_brain(W,G,noise,heart,energy,seed):
    rng = np.random.default_rng(seed)
    N = W.shape[0]
    z = rng.normal(size=N) + 1j*rng.normal(size=N)
    omega = rng.uniform(0.04,0.07,N)
    row = W.sum(axis=1)
    traj_power = np.zeros((STEPS,N))
    for t in range(STEPS):
        coupling = G*(W@z - row*z)
        dz = (0.02 + 1j*omega - np.abs(z)**2)*z + coupling
        dz += K_BODY*heart[t] + K_ENERGY*energy[t]
        dz += noise*np.sqrt(DT)*(rng.normal(size=N)+1j*rng.normal(size=N))
        z += dz*DT
        traj_power[t] = np.abs(z)**2
    return traj_power

# ============================================================
# METRICS
# ============================================================
def metastability(traj_power):
    phase = np.angle(traj_power.astype(complex))
    R = np.abs(np.mean(np.exp(1j*phase),axis=1))
    return np.std(R)

def coherence(sig1,sig2):
    phase1 = np.angle(hilbert(sig1))
    phase2 = np.angle(hilbert(sig2))
    return np.abs(np.mean(np.exp(1j*(phase1-phase2))))

def predictive_metric(traj_power):
    amp = traj_power
    diff = np.diff(np.mean(amp,axis=1))
    return np.sqrt(np.mean(diff**2))

def energy_metrics(E):
    return np.std(E), np.mean(E)

# ============================================================
# SIMULATE SUBJECT
# ============================================================
def simulate_subject(W, network_id, seed, subject_id):
    rng = np.random.default_rng(seed + subject_id)
    age = rng.uniform(20,80)
    G = rng.uniform(0.2,1.0)
    noise = rng.uniform(0.01,0.05)
    heart, resp = simulate_body(seed + subject_id)
    brain_power_dummy = np.ones(STEPS)
    energy_dummy = simulate_energy(brain_power_dummy, heart, resp)
    traj_power = simulate_brain(W,G,noise,heart,energy_dummy,seed + subject_id)
    brain_global = np.mean(traj_power,axis=1)
    m = metastability(traj_power)
    coh = coherence(brain_global, heart)
    pred = predictive_metric(traj_power)
    brain_power = np.mean(traj_power,axis=1)
    energy = simulate_energy(brain_power, heart, resp)
    stab, eff = energy_metrics(energy)
    return {
        "NetworkID": network_id,
        "Seed": seed,
        "SubjectID": subject_id,
        "Age": age,
        "G": G,
        "Noise": noise,
        "Metastability": m,
        "BrainHeartCoherence": coh,
        "PredictiveCoding": pred,
        "EnergyStability": stab,
        "EnergyEfficiency": eff
    }

# ============================================================
# GENERATE DATASET PARALLEL
# ============================================================
def generate_dataset_parallel():
    tasks = []
    network_counter = 0
    for N in NETWORK_SIZES:
        for net_idx in range(N_NETWORKS_PER_SIZE):
            W = small_world(N, seed=net_idx)
            for seed in range(SEEDS_PER_NETWORK):
                for subj in range(N_SUBJECTS_PER_SEED):
                    tasks.append((W,network_counter,seed,subj))
            network_counter += 1
    results = Parallel(n_jobs=-1)(delayed(simulate_subject)(*t) for t in tasks)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(SAVE_DIR,"dataset.csv"),index=False)
    return df

# ============================================================
# CROSS-VALIDATED R² AND EFFECT SIZE
# ============================================================
def crossval_r2(df,predictor,target):
    X = df[[predictor]].values
    y = df[target].values
    kf = KFold(10,shuffle=True,random_state=42)
    r2_list=[]
    for tr,te in kf.split(X):
        model = LinearRegression().fit(X[tr],y[tr])
        pred_vals = model.predict(X[te])
        r,_ = pearsonr(pred_vals,y[te])
        r2_list.append(r**2)
    mean_r2 = np.mean(r2_list)
    ci_lower = np.percentile(r2_list,2.5)
    ci_upper = np.percentile(r2_list,97.5)
    return mean_r2,(ci_lower,ci_upper),r2_list

def permutation_test(df,predictor,target,n_perm=N_PERM):
    real_r2, ci, _ = crossval_r2(df,predictor,target)
    perm_r2=[]
    for _ in range(n_perm):
        df_perm = df.copy()
        df_perm[predictor] = np.random.permutation(df[predictor].values)
        r2,_,_ = crossval_r2(df_perm,predictor,target)
        perm_r2.append(r2)
    p_val = np.mean(np.array(perm_r2)>=real_r2)
    return real_r2, ci, p_val, perm_r2

# ============================================================
# RUN SIMULATION
# ============================================================
print("Generating dataset...")
df = generate_dataset_parallel()
print(f"Dataset saved: {df.shape[0]} subjects")

# ============================================================
# MODEL COMPARISON
# ============================================================
predictors = ["Metastability","BrainHeartCoherence","PredictiveCoding"]
targets = ["EnergyStability","EnergyEfficiency"]
results=[]
perm_distributions={}

for pred,targ in product(predictors,targets):
    r2, ci, pval, perm_vals = permutation_test(df,pred,targ)
    results.append({
        "Predictor":pred,
        "Target":targ,
        "CrossVal_R2":r2,
        "CI_lower":ci[0],
        "CI_upper":ci[1],
        "p_value":pval
    })
    perm_distributions[(pred,targ)] = perm_vals

res_df = pd.DataFrame(results)
res_df["p_adj_bonf"] = multipletests(res_df["p_value"],method='bonferroni')[1]
res_df["p_adj_fdr"] = multipletests(res_df["p_value"],method='fdr_bh')[1]
res_df.to_csv(os.path.join(SAVE_DIR,"model_comparison_table.csv"),index=False)
print("Model comparison table saved.")

# ============================================================
# FIGURES
# ============================================================

sns.set(style="whitegrid")

# CV R² barplot with CI
plt.figure(figsize=(10,6))
sns.barplot(data=res_df,x="Predictor",y="CrossVal_R2",hue="Target",ci=None,palette="Set2")
for i,row in res_df.iterrows():
    plt.errorbar(x=i%len(predictors),y=row["CrossVal_R2"],
                 yerr=[[row["CrossVal_R2"]-row["CI_lower"]],[row["CI_upper"]-row["CrossVal_R2"]]],
                 fmt='none',c='black',capsize=5)
plt.ylabel("Cross-validated R²")
plt.title("Predictors of Energy Stability and Efficiency")
plt.legend(title="Target")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR,"model_comparison_CV_R2.png"),dpi=300)
plt.show()

# Permutation histograms
for pred,targ in product(predictors,targets):
    perm_vals = perm_distributions[(pred,targ)]
    r2_real = res_df.loc[(res_df["Predictor"]==pred)&(res_df["Target"]==targ),"CrossVal_R2"].values[0]
    plt.figure(figsize=(6,4))
    sns.histplot(perm_vals,kde=True,color='skyblue',bins=20)
    plt.axvline(r2_real,color='red',linestyle='--',label='Observed R²')
    plt.xlabel("R²")
    plt.ylabel("Count")
    plt.title(f"Permutation Test: {pred} → {targ}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR,f"perm_hist_{pred}_{targ}.png"),dpi=300)
    plt.show()

# Predictor correlation heatmap
plt.figure(figsize=(6,5))
sns.heatmap(df[predictors].corr(),annot=True,cmap="coolwarm")
plt.title("Predictor Correlations")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR,"predictor_correlation.png"),dpi=300)
plt.show()

# Age vs Energy Efficiency/Stability regression
for targ in targets:
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df,x="Age",y=targ,alpha=0.3)
    sns.regplot(data=df,x="Age",y=targ,scatter=False,color='red')
    plt.title(f"Age vs {targ}")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR,f"age_{targ}.png"),dpi=300)
    plt.show()

print("Simulation complete. All results saved in:", SAVE_DIR)
