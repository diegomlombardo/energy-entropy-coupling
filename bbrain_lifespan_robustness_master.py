# ============================================================
# Brain-Body-Energy Generative Model - Publication Figures
# ============================================================

import numpy as np
import pandas as pd
import json
from scipy.signal import hilbert
from scipy.stats import pearsonr, zscore
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ===========================
# GLOBAL PARAMETERS
# ===========================
SEED_GLOBAL = 42
rng_global = np.random.default_rng(SEED_GLOBAL)

NETWORK_SIZES = [40, 60, 80, 100, 120]
TOPOLOGIES = ["small_world", "random", "fully_connected"]
N_SEEDS = 50
N_SUBJECTS = 300
T = 500
DT = 0.05
STEPS = int(T / DT)
N_PERM = 200
K_BODY = 0.05
K_ENERGY = 0.05

SAVE_DIR = "BrainBodyEnergy_PublicationFigures"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===========================
# LOGGING
# ===========================
def log(msg): print(f"[INFO] {msg}")

# ===========================
# NETWORK GENERATION
# ===========================
def generate_network(N, topology, seed=None):
    rng = np.random.default_rng(seed)
    if topology == "small_world":
        W = np.zeros((N, N))
        k = max(2, N//10)
        p = 0.2
        for i in range(N):
            for j in range(1, k//2 + 1):
                W[i, (i+j)%N] = 1
                W[i, (i-j)%N] = 1
        for i in range(N):
            for j in range(N):
                if W[i,j]==1 and rng.random()<p:
                    W[i,j]=0
                    new_j = rng.integers(N)
                    while new_j==i or W[i,new_j]==1:
                        new_j = rng.integers(N)
                    W[i,new_j]=1
        W = (W + W.T)/2
        np.fill_diagonal(W, 0)
        W = W/np.max(np.abs(np.linalg.eigvals(W)))
    elif topology == "random":
        W = rng.random((N,N))
        W = (W + W.T)/2
        np.fill_diagonal(W,0)
        W = W/np.max(np.abs(np.linalg.eigvals(W)))
    elif topology == "fully_connected":
        W = np.ones((N,N)) - np.eye(N)
        W = W/np.max(np.abs(np.linalg.eigvals(W)))
    return W

# ===========================
# BODY OSCILLATORS
# ===========================
def simulate_body(seed):
    rng = np.random.default_rng(seed)
    heart_phase, resp_phase = 0,0
    heart, resp = [],[]
    for t in range(STEPS):
        heart_phase += 0.9*DT + rng.normal(scale=0.01)
        resp_phase += 0.25*DT + rng.normal(scale=0.01)
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
        cons = 0.05*brain_power[t]
        dE = prod - cons - decay*E
        E += dE*DT
        E_series.append(E)
    return np.array(E_series)

# ===========================
# BRAIN DYNAMICS: Free-Energy
# ===========================
def simulate_brain_FE(W,G,noise,heart,energy,seed):
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
        dz += noise*np.sqrt(DT)*(rng.normal(size=N) + 1j*rng.normal(size=N))
        z += dz*DT
        traj_power[t] = np.abs(z)**2
    return traj_power

# ===========================
# BRAIN DYNAMICS: Wilson–Cowan
# ===========================
def simulate_brain_WC(W, seed):
    rng = np.random.default_rng(seed)
    N = W.shape[0]
    dt = 0.05
    Tsteps = STEPS
    E = rng.random(N)
    I = rng.random(N)
    a=1.0; b=1.0; c=1.0
    traj = np.zeros((Tsteps,N))
    for t in range(Tsteps):
        dE = -E + np.tanh(a*E - b*I + W@E)
        dI = -I + np.tanh(c*E)
        E += dE*dt
        I += dI*dt
        traj[t] = E
    return traj

# ===========================
# METRICS
# ===========================
def metastability(traj_power):
    phase = np.angle(traj_power.astype(complex))
    R = np.abs(np.mean(np.exp(1j*phase), axis=1))
    return np.std(R)

def coherence(sig1,sig2):
    phase1 = np.angle(hilbert(sig1))
    phase2 = np.angle(hilbert(sig2))
    return np.abs(np.mean(np.exp(1j*(phase1-phase2))))

def energy_metrics(E):
    return np.std(E), np.mean(E)

def generate_cognition(traj_power, seed):
    rng = np.random.default_rng(seed)
    m = metastability(traj_power)
    ph_var = np.var(np.angle(traj_power.astype(complex)), axis=0).mean()
    cog = 0.5*m + 0.5*ph_var + rng.normal(scale=0.05)
    return cog

# ===========================
# SIMULATE SUBJECT
# ===========================
def simulate_subject(W, network_size, topology, seed, subject_id):
    rng = np.random.default_rng(seed+subject_id)
    age = rng.uniform(20,80)
    G = rng.uniform(0.2,1.0)
    noise = rng.uniform(0.01,0.05)
    heart, resp = simulate_body(seed+subject_id)
    brain_power_dummy = np.ones(STEPS)
    energy_dummy = simulate_energy(brain_power_dummy,heart,resp)
    
    # Models
    traj_FE = simulate_brain_FE(W,G,noise,heart,energy_dummy,seed+subject_id)
    traj_WC = simulate_brain_WC(W, seed+subject_id)
    traj_null = rng.normal(size=(STEPS,W.shape[0]))
    
    brain_FE = traj_FE.mean(axis=1)
    brain_WC = traj_WC.mean(axis=1)
    brain_null = traj_null.mean(axis=1)
    
    m_FE = metastability(traj_FE)
    coh_FE = coherence(brain_FE,heart)
    E_FE = simulate_energy(brain_FE,heart,resp)
    stab_FE, eff_FE = energy_metrics(E_FE)
    cog_FE = generate_cognition(traj_FE, seed+subject_id)
    
    m_WC = metastability(traj_WC)
    coh_WC = coherence(brain_WC,heart)
    E_WC = simulate_energy(brain_WC,heart,resp)
    stab_WC, eff_WC = energy_metrics(E_WC)
    cog_WC = generate_cognition(traj_WC, seed+subject_id)
    
    m_null = metastability(traj_null)
    coh_null = coherence(brain_null,heart)
    E_null = simulate_energy(brain_null,heart,resp)
    stab_null, eff_null = energy_metrics(E_null)
    cog_null = generate_cognition(traj_null, seed+subject_id)
    
    return {
        "Seed": seed,
        "NetworkSize": network_size,
        "Topology": topology,
        "Age": age,
        "SubjectID": subject_id,
        "Metastability_FE": m_FE, "Coherence_FE": coh_FE, "EnergyStability_FE": stab_FE, "EnergyEfficiency_FE": eff_FE, "Cognition_FE": cog_FE,
        "Metastability_WC": m_WC, "Coherence_WC": coh_WC, "EnergyStability_WC": stab_WC, "EnergyEfficiency_WC": eff_WC, "Cognition_WC": cog_WC,
        "Metastability_Null": m_null, "Coherence_Null": coh_null, "EnergyStability_Null": stab_null, "EnergyEfficiency_Null": eff_null, "Cognition_Null": cog_null,
    }

# ===========================
# DATASET GENERATION
# ===========================
def generate_dataset():
    all_results=[]
    log("Starting dataset generation...")
    for size in NETWORK_SIZES:
        for topo in TOPOLOGIES:
            log(f"Generating networks: size {size}, topology {topo}")
            for seed in range(N_SEEDS):
                W = generate_network(size, topo, seed)
                results = Parallel(n_jobs=-1)(delayed(simulate_subject)(W,size,topo,seed,s) for s in range(N_SUBJECTS))
                all_results.extend(results)
                log(f"Completed seed {seed} for size {size} topo {topo}")
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(SAVE_DIR,"dataset_full.csv"),index=False)
    log("Dataset generation complete")
    return df

# ===========================
# MEDIATION FUNCTION
# ===========================
def simple_mediation(df, X, M, Y):
    """Computes simple mediation path X -> M -> Y."""
    # Standardize
    df_std = df[[X,M,Y]].apply(zscore)
    # Regression X->M
    reg1 = LinearRegression().fit(df_std[[X]], df_std[M])
    a = reg1.coef_[0]
    # Regression M->Y controlling for X
    reg2 = LinearRegression().fit(df_std[[M,X]], df_std[Y])
    b = reg2.coef_[0]
    direct = reg2.coef_[1]
    indirect = a*b
    return {"a":a,"b":b,"direct":direct,"indirect":indirect}

# ===========================
# PLOTTING FUNCTION
# ===========================
def plot_corr_trends(df, metric, cog, ylabel, title):
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=metric, y=cog, hue='NetworkSize', style='Topology', data=df, alpha=0.3)
    sns.lineplot(x=metric, y=cog, hue='NetworkSize', style='Topology', data=df, ci='sd')
    plt.xlabel(metric)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def plot_age_trends(df, cog, title):
    plt.figure(figsize=(10,6))
    sns.scatterplot(x="Age", y=cog, hue='Topology', style='NetworkSize', data=df, alpha=0.3)
    sns.lineplot(x="Age", y=cog, hue='Topology', style='NetworkSize', data=df, ci='sd')
    plt.xlabel("Age")
    plt.ylabel("Cognition")
    plt.title(title)
    plt.show()

# ===========================
# RUN SIMULATION
# ===========================
if __name__=="__main__":
    df = generate_dataset()
    
    # ---- Correlation plots ----
    plot_corr_trends(df,"EnergyEfficiency_FE","Cognition_FE","Cognition","FE Model: Cognition vs Energy Efficiency")
    plot_corr_trends(df,"EnergyEfficiency_WC","Cognition_WC","Cognition","WC Model: Cognition vs Energy Efficiency")
    
    # ---- Mediation example ----
    med_FE = simple_mediation(df,"Coherence_FE","EnergyEfficiency_FE","Cognition_FE")
    med_WC = simple_mediation(df,"Coherence_WC","EnergyEfficiency_WC","Cognition_WC")
    print("Mediation FE model:", med_FE)
    print("Mediation WC model:", med_WC)
    
    # ---- Lifespan trends ----
    plot_age_trends(df,"Cognition_FE","FE Cognition across lifespan")
    plot_age_trends(df,"Cognition_WC","WC Cognition across lifespan")
    
    # ---- Summary Table ----
    summary = pd.DataFrame({
        "Model":["FE","WC"],
        "Energy->Cognition r":[pearsonr(df["EnergyEfficiency_FE"],df["Cognition_FE"])[0],
                               pearsonr(df["EnergyEfficiency_WC"],df["Cognition_WC"])[0]],
        "Indirect a*b":[med_FE["indirect"],med_WC["indirect"]],
        "Direct effect":[med_FE["direct"],med_WC["direct"]]
    })
    print("Summary Table:")
    print(summary)
