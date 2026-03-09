# ============================================================
# BRAIN–BODY–ENERGY IN SILICO MODEL – THREE MODELS COMPARISON
# Brain–Body, Feedforward Free-Energy, Null
# ============================================================

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# -----------------------------
# GLOBAL PARAMETERS
# -----------------------------
SEED_GLOBAL = 42
rng_global = np.random.default_rng(SEED_GLOBAL)

NETWORK_SIZES = [40, 80, 120]
N_NETWORKS_PER_SIZE = 5
SEEDS_PER_NETWORK = 5
N_SUBJECTS_PER_SEED = 50
T = 200
DT = 0.05
STEPS = int(T/DT)
K_BODY_LIST = [0.02, 0.04, 0.06]

SAVE_DIR = "BrainBodyEnergy_ThreeModels"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# NETWORK GENERATION
# -----------------------------
def small_world_network(N, k=6, p=0.2, seed=0):
    rng = np.random.default_rng(seed)
    W = np.zeros((N,N))
    for i in range(N):
        for j in range(1,k//2+1):
            W[i,(i+j)%N] = 1
            W[i,(i-j)%N] = 1
    for i in range(N):
        for j in range(N):
            if W[i,j]==1 and rng.random()<p:
                W[i,j]=0
                new=rng.integers(N)
                while new==i or W[i,new]==1:
                    new=rng.integers(N)
                W[i,new]=1
    W=(W+W.T)/2
    np.fill_diagonal(W,0)
    eig=np.linalg.eigvals(W)
    W/=np.max(np.abs(eig))
    return W

# ============================================================
# TRAITS
# -----------------------------
def sample_traits(rng):
    return {
        "autonomic": rng.normal(),
        "metabolic": rng.normal(),
        "neural_gain": rng.normal()
    }

# ============================================================
# BODY OSCILLATIONS
# -----------------------------
def simulate_body(traits, rng):
    heart_phase, resp_phase = 0,0
    heart, resp = [], []
    hr_base = 0.9 + 0.05*traits["autonomic"]
    rr_base = 0.25 + 0.03*traits["autonomic"]
    for t in range(STEPS):
        heart_phase += hr_base*DT + rng.normal(scale=0.01)
        resp_phase += rr_base*DT + rng.normal(scale=0.01)
        heart.append(np.sin(heart_phase))
        resp.append(np.sin(resp_phase))
    return np.array(heart), np.array(resp)

# ============================================================
# BRAIN DYNAMICS
# -----------------------------
def simulate_brain(W, traits, heart, K_BODY, rng):
    N = W.shape[0]
    z = rng.normal(size=N)+1j*rng.normal(size=N)
    omega = rng.uniform(0.04,0.07,N)
    gain = 0.02 + 0.01*traits["neural_gain"]
    phases = np.zeros((STEPS,N))
    power = np.zeros((STEPS,N))
    R_t = np.zeros(STEPS)
    row_sum = W.sum(axis=1)
    for t in range(STEPS):
        coupling = 0.6*(W @ z - row_sum*z)
        dz = (gain + 1j*omega - np.abs(z)**2)*z + coupling
        dz += K_BODY*heart[t]
        z += dz*DT + 0.02*np.sqrt(DT)*(rng.normal(size=N)+1j*rng.normal(size=N))
        phases[t] = np.angle(z)
        power[t] = np.abs(z)**2
        R_t[t] = np.abs(np.mean(np.exp(1j*phases[t])))
    metastab = np.std(R_t)
    return phases, power, R_t, metastab

# ============================================================
# ENERGY MODELS
# -----------------------------
def simulate_energy_ff(brain_power, heart, resp, traits):
    E = 1
    series=[]
    decay=0.02
    for t in range(STEPS):
        production = 0.4 + 0.2*heart[t] + 0.1*resp[t] + 0.1*traits["metabolic"]
        consumption = 0.05*brain_power[t] + 0.02*np.mean(brain_power[:t+1])
        dE = production - consumption - decay*E
        E += dE*DT
        series.append(E)
    return np.array(series)

def simulate_energy_null(brain_power, heart, resp, traits):
    E = 1
    series=[]
    decay=0.02
    for t in range(STEPS):
        production = 0.4 + 0.2*heart[t] + 0.1*resp[t] + 0.1*traits["metabolic"]
        consumption = 0.05 + 0.01*rng_global.normal()
        dE = production - consumption - decay*E
        E += dE*DT
        series.append(E)
    return np.array(series)

def simulate_energy_brainbody(coh, brain_power, heart, resp, traits):
    # Vulnerability model: Brain–Body coherence drives energy
    E = 1
    series=[]
    decay=0.02
    for t in range(STEPS):
        production = 0.4 + 0.2*heart[t] + 0.1*resp[t] + 0.1*traits["metabolic"]
        consumption = 0.05*brain_power[t] + 0.05*coh
        dE = production - consumption - decay*E
        E += dE*DT
        series.append(E)
    return np.array(series)

# ============================================================
# METRICS
# -----------------------------
def brain_body_coherence(brain, heart):
    phase1 = np.angle(np.exp(1j*brain))
    phase2 = np.angle(np.exp(1j*heart))
    return np.abs(np.mean(np.exp(1j*(phase1-phase2))))

def predictive_complexity(power):
    signal = power.mean(axis=1)
    diff = np.diff(signal)
    return np.sqrt(np.mean(diff**2))

def energy_metrics(E):
    return np.std(E), np.mean(E)

def generate_cognition(traits):
    return 0.5*traits["neural_gain"] + 0.4*traits["metabolic"] + 0.2*rng_global.normal()

# ============================================================
# SIMULATE ONE SUBJECT
# -----------------------------
def simulate_subject(W, K_BODY, seed_sub):
    rng = np.random.default_rng(seed_sub)
    traits = sample_traits(rng)
    heart, resp = simulate_body(traits, rng)
    phases, power, R_t, metastab = simulate_brain(W, traits, heart, K_BODY, rng)
    brain_global = power.mean(axis=1)
    coh = brain_body_coherence(brain_global, heart)
    comp = predictive_complexity(power)
    # Energy for three models
    energy_ff = simulate_energy_ff(brain_global, heart, resp, traits)
    energy_null = simulate_energy_null(brain_global, heart, resp, traits)
    energy_vul = simulate_energy_brainbody(coh, brain_global, heart, resp, traits)
    stab_ff, eff_ff = energy_metrics(energy_ff)
    stab_null, eff_null = energy_metrics(energy_null)
    stab_vul, eff_vul = energy_metrics(energy_vul)
    cog = generate_cognition(traits)
    return {
        "Metastability": metastab,
        "BrainHeartCoherence": coh,
        "ForwardModeling": comp,
        "EnergyStability_FF": stab_ff,
        "EnergyEfficiency_FF": eff_ff,
        "EnergyStability_Null": stab_null,
        "EnergyEfficiency_Null": eff_null,
        "EnergyStability_Vul": stab_vul,
        "EnergyEfficiency_Vul": eff_vul,
        "Cognition": cog
    }

# ============================================================
# PARALLEL DATASET GENERATION
# -----------------------------
def generate_dataset_full():
    all_results=[]
    total_tasks = len(NETWORK_SIZES)*N_NETWORKS_PER_SIZE*SEEDS_PER_NETWORK*len(K_BODY_LIST)
    task_counter = 0
    start_time = time.time()
    for N in NETWORK_SIZES:
        for net_idx in range(N_NETWORKS_PER_SIZE):
            W = small_world_network(N, seed=net_idx)
            for seed_net in range(SEEDS_PER_NETWORK):
                for K_BODY_val in K_BODY_LIST:
                    results = Parallel(n_jobs=-1)(
                        delayed(simulate_subject)(W, K_BODY_val, seed_net*1000 + subj)
                        for subj in range(N_SUBJECTS_PER_SEED)
                    )
                    df_seed = pd.DataFrame(results)
                    df_seed["NetworkSize"] = N
                    df_seed["K_BODY"] = K_BODY_val
                    df_seed["SeedNetwork"] = seed_net
                    all_results.append(df_seed)
                    task_counter += 1
                    print(f"[{task_counter}/{total_tasks}] N={N} net={net_idx} seed={seed_net} K_BODY={K_BODY_val}, elapsed {time.time()-start_time:.1f}s")
    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv(os.path.join(SAVE_DIR,"dataset_full_three_models.csv"), index=False)
    return df_all

# ============================================================
# CROSS-VALIDATION & PERMUTATION
# -----------------------------
def crossval_r2(df, predictor, target):
    X = df[[predictor]].values
    y = df[target].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    kf = KFold(10, shuffle=True, random_state=SEED_GLOBAL)
    r2_scores = []
    for tr, te in kf.split(X):
        model = LinearRegression().fit(X[tr], y[tr])
        pred = model.predict(X[test])
        r2_scores.append(r2_score(y[test], pred))
    return np.mean(r2_scores), np.std(r2_scores)

def permutation_test(df, predictor, target, n_perm=100):
    real, _ = crossval_r2(df, predictor, target)
    null = []
    for _ in range(n_perm):
        df_perm = df.copy()
        df_perm[target] = rng_global.permutation(df_perm[target])
        r2, _ = crossval_r2(df_perm, predictor, target)
        null.append(r2)
    p_val = np.mean(np.array(null) >= real)
    return real, p_val

# ============================================================
# MEDIATION: Brain–Body Coherence → Energy → Cognition
# -----------------------------
def mediation_analysis(df, energy_col):
    df_std = df.copy()
    for col in ["Cognition","BrainHeartCoherence", energy_col]:
        df_std[col] = (df_std[col]-df_std[col].mean())/df_std[col].std()
    med_model = smf.ols(f'{energy_col} ~ BrainHeartCoherence', data=df_std).fit()
    out_model = smf.ols(f'Cognition ~ {energy_col} + BrainHeartCoherence', data=df_std).fit()
    indirect = med_model.params["BrainHeartCoherence"]*out_model.params[energy_col]
    direct = out_model.params["BrainHeartCoherence"]
    total = direct + indirect
    return {"Direct":direct, "Indirect":indirect, "Total":total}

# ============================================================
# RUN FULL SIMULATION
# -----------------------------
print("Starting full sweep (6–12h expected)...")
df = generate_dataset_full()

# SUMMARY TABLE
summary_table = df[[
    "BrainHeartCoherence",
    "EnergyEfficiency_FF","EnergyEfficiency_Null","EnergyEfficiency_Vul",
    "Cognition"
]].describe()
print("\nSUMMARY TABLE")
print(summary_table)

# CORRELATION MATRIX
corr_matrix = df[["BrainHeartCoherence","EnergyEfficiency_FF","EnergyEfficiency_Null","EnergyEfficiency_Vul","Cognition"]].corr()
print("\nCORRELATION MATRIX")
print(corr_matrix)

# CROSS-VALIDATION
r2_ff, std_ff = crossval_r2(df,"EnergyEfficiency_FF","Cognition")
r2_null, std_null = crossval_r2(df,"EnergyEfficiency_Null","Cognition")
r2_vul, std_vul = crossval_r2(df,"EnergyEfficiency_Vul","Cognition")
print(f"\nCross-validated R² FF: {r2_ff:.3f} ± {std_ff:.3f}")
print(f"Cross-validated R² Null: {r2_null:.3f} ± {std_null:.3f}")
print(f"Cross-validated R² Vul: {r2_vul:.3f} ± {std_vul:.3f}")

# PERMUTATION TEST
r2_ff_perm, p_ff = permutation_test(df,"EnergyEfficiency_FF","Cognition")
r2_null_perm, p_null = permutation_test(df,"EnergyEfficiency_Null","Cognition")
r2_vul_perm, p_vul = permutation_test(df,"EnergyEfficiency_Vul","Cognition")
print(f"Permutation test FF: R²={r2_ff_perm:.3f}, p={p_ff:.3f}")
print(f"Permutation test Null: R²={r2_null_perm:.3f}, p={p_null:.3f}")
print(f"Permutation test Vul: R²={r2_vul_perm:.3f}, p={p_vul:.3f}")

# MEDIATION
med_ff = mediation_analysis(df, "EnergyEfficiency_FF")
med_null = mediation_analysis(df, "EnergyEfficiency_Null")
med_vul = mediation_analysis(df, "EnergyEfficiency_Vul")
print("\nMEDIATION RESULTS")
print("Feedforward model:", med_ff)
print("Null model:", med_null)
print("Brain-Body Vulnerability model:", med_vul)

# FIGURES
plt.figure(figsize=(8,5))
sns.histplot(df["Cognition"], bins=50, kde=True, color="skyblue")
plt.title("Distribution of Cognition")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x="EnergyEfficiency_FF", y="Cognition", data=df, alpha=0.5, label="Feedforward")
sns.scatterplot(x="EnergyEfficiency_Null", y="Cognition", data=df, alpha=0.5, label="Null")
sns.scatterplot(x="EnergyEfficiency_Vul", y="Cognition", data=df, alpha=0.5, label="Vulnerability")
plt.title("Energy Efficiency vs Cognition (Three Models)")
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix Key Metrics")
plt.show()
