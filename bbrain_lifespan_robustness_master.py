# ============================================================
# Brain–Body–Energy Generative Model
# Publication-Grade Computational Framework
# ============================================================

import numpy as np
import pandas as pd
import os
import json
from scipy.signal import hilbert
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# ============================================================
# GLOBAL PARAMETERS
# ============================================================

SEED = 42
rng = np.random.default_rng(SEED)

NETWORK_SIZES = [40, 80, 120]
N_NETWORKS_PER_SIZE = 5
SUBJECTS_PER_NETWORK = 200

T = 400
DT = 0.05
STEPS = int(T / DT)

SAVE_DIR = "BrainBodyEnergy_Final"
FIG_DIR = os.path.join(SAVE_DIR, "figures")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# NETWORK GENERATION
# ============================================================

def small_world(N, k=6, p=0.2, seed=None):

    rng = np.random.default_rng(seed)

    W = np.zeros((N,N))

    for i in range(N):
        for j in range(1, k//2+1):
            W[i,(i+j)%N] = 1
            W[i,(i-j)%N] = 1

    for i in range(N):
        for j in range(N):
            if W[i,j] == 1 and rng.random() < p:
                W[i,j] = 0
                new = rng.integers(N)
                while new == i:
                    new = rng.integers(N)
                W[i,new] = 1

    W = (W + W.T)/2
    np.fill_diagonal(W,0)

    eig = np.linalg.eigvals(W)
    W /= np.max(np.abs(eig))

    return W

# ============================================================
# BODY DYNAMICS
# ============================================================

def simulate_body():

    heart_phase = 0
    resp_phase = 0

    heart = np.zeros(STEPS)
    resp = np.zeros(STEPS)

    for t in range(STEPS):

        resp_phase += 0.25*DT + rng.normal(0,0.005)

        heart_rate = 1.0 + 0.15*np.sin(resp_phase)

        heart_phase += heart_rate*DT + rng.normal(0,0.01)

        heart[t] = np.sin(heart_phase)
        resp[t] = np.sin(resp_phase)

    return heart, resp

# ============================================================
# BRAIN + ENERGY COUPLED DYNAMICS
# ============================================================

def simulate_brain_energy(W):

    N = W.shape[0]

    z = rng.normal(size=N) + 1j*rng.normal(size=N)

    omega = rng.uniform(0.04,0.07,N)

    heart, resp = simulate_body()

    node_body_weights = rng.normal(0.5,0.1,N)

    energy = 1.0
    energy_series = np.zeros(STEPS)

    phase_traj = np.zeros((STEPS,N))

    brain_power = np.zeros(STEPS)

    for t in range(STEPS):

        coupling = 0.5*(W @ z - W.sum(axis=1)*z)

        body_input = node_body_weights*(0.6*heart[t] + 0.3*resp[t])

        dz = (0.02 + 1j*omega - np.abs(z)**2)*z
        dz += coupling
        dz += 0.03*energy
        dz += body_input

        dz += 0.02*(rng.normal(size=N)+1j*rng.normal(size=N))

        z += dz*DT

        phase_traj[t] = np.angle(z)

        power = np.mean(np.abs(z)**2)

        brain_power[t] = power

        prod = 0.4 + 0.1*heart[t] + 0.1*resp[t]

        cons = 0.08*power

        dE = prod - cons - 0.03*energy

        energy += dE*DT

        energy = max(0.05, min(3, energy))

        energy_series[t] = energy

    return phase_traj, brain_power, energy_series, heart, resp

# ============================================================
# METRICS
# ============================================================

def metastability(phase):

    R = np.abs(np.mean(np.exp(1j*phase),axis=1))

    return np.std(R)

def brain_body_coherence(brain, heart):

    p1 = np.angle(hilbert(brain))
    p2 = np.angle(hilbert(heart))

    return np.abs(np.mean(np.exp(1j*(p1-p2))))

def predictive_complexity(signal):

    diff = np.diff(signal)

    return np.sqrt(np.mean(diff**2))

# ============================================================
# SYNTHETIC COGNITION
# ============================================================

def generate_cognition(brain_power, metastab):

    latent = 0.6*np.tanh(np.mean(brain_power)) + 0.4*metastab

    return latent + rng.normal(0,0.05)

# ============================================================
# SUBJECT SIMULATION
# ============================================================

def simulate_subject(W, net_id, subj_id):

    phase, power, energy, heart, resp = simulate_brain_energy(W)

    m = metastability(phase)

    coh = brain_body_coherence(power, heart)

    pred = predictive_complexity(power)

    cog = generate_cognition(power, m)

    return {

        "NetworkID":net_id,
        "SubjectID":subj_id,

        "Metastability":m,
        "BrainHeartCoherence":coh,
        "ForwardModeling":pred,

        "EnergyMean":np.mean(energy),
        "EnergyStd":np.std(energy),

        "Cognition":cog
    }

# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset():

    all_data = []

    net_id = 0

    for N in NETWORK_SIZES:

        for n in range(N_NETWORKS_PER_SIZE):

            W = small_world(N, seed=n)

            results = Parallel(n_jobs=-1)(
                delayed(simulate_subject)(W,net_id,s)
                for s in range(SUBJECTS_PER_NETWORK)
            )

            all_data.extend(results)

            net_id += 1

    df = pd.DataFrame(all_data)

    df.to_csv(os.path.join(SAVE_DIR,"dataset.csv"),index=False)

    return df

# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

def crossval(df, predictor):

    X = df[[predictor]].values
    y = df["Cognition"].values
    groups = df["NetworkID"]

    gkf = GroupKFold(10)

    scores = []

    for tr,te in gkf.split(X,y,groups):

        model = LinearRegression()

        model.fit(X[tr],y[tr])

        pred = model.predict(X[te])

        scores.append(r2_score(y[te],pred))

    return np.mean(scores), np.std(scores)

# ============================================================
# FIGURES
# ============================================================

def create_figures(df):

    sns.set(style="whitegrid",context="talk")

    predictors = [
        "Metastability",
        "BrainHeartCoherence",
        "ForwardModeling"
    ]

    fig,ax = plt.subplots(1,3,figsize=(18,5))

    for i,p in enumerate(predictors):

        sns.regplot(
            x=p,
            y="Cognition",
            data=df,
            scatter_kws={"alpha":0.3},
            line_kws={"color":"black"},
            ax=ax[i]
        )

    plt.tight_layout()

    plt.savefig(f"{FIG_DIR}/Predictors_vs_Cognition.png",dpi=600)

    plt.close()

    plt.figure(figsize=(6,5))

    sns.histplot(df["EnergyMean"],bins=40,kde=True)

    plt.title("Energy Distribution")

    plt.tight_layout()

    plt.savefig(f"{FIG_DIR}/Energy_Distribution.png",dpi=600)

# ============================================================
# SUMMARY TABLE
# ============================================================

def summary_table(df):

    predictors = [
        "Metastability",
        "BrainHeartCoherence",
        "ForwardModeling"
    ]

    rows = []

    for p in predictors:

        r,_ = pearsonr(df[p],df["Cognition"])

        r2,std = crossval(df,p)

        rows.append({
            "Predictor":p,
            "Correlation":r,
            "CrossVal_R2":r2,
            "CV_SD":std
        })

    table = pd.DataFrame(rows)

    table.to_csv(os.path.join(SAVE_DIR,"summary_statistics.csv"),index=False)

    return table

# ============================================================
# RUN PIPELINE
# ============================================================

print("Generating dataset...")

df = generate_dataset()

print("Creating figures...")

create_figures(df)

print("Generating summary table...")

table = summary_table(df)

print(table)

print("Pipeline complete.")
