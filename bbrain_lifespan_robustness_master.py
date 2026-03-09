# ============================================================
# ROBUST BRAIN–BODY–ENERGY IN SILICO MODEL
# WITH MODEL COMPARISON AND MEDIATION
# ============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import networkx as nx

# ============================================================
# GLOBAL PARAMETERS
# ============================================================

SEED_GLOBAL = 42
rng_global = np.random.default_rng(SEED_GLOBAL)

NETWORK_SIZES = [40, 80]
N_NETWORKS_PER_SIZE = 3
SEEDS_PER_NETWORK = 10
SUBJECTS_PER_SEED = 50

T = 200
DT = 0.05
STEPS = int(T / DT)
K_BODY = 0.04

# ============================================================
# NETWORK GENERATION
# ============================================================

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
# TRAITS AND BODY OSCILLATIONS
# ============================================================

def sample_traits(seed=None):
    rng = np.random.default_rng(seed)
    return {
        "autonomic": rng.normal(),
        "metabolic": rng.normal(),
        "neural_gain": rng.normal()
    }

def simulate_body(traits):
    heart_phase = 0
    resp_phase = 0
    heart = []
    resp = []
    hr_base = 0.9 + 0.05*traits["autonomic"]
    rr_base = 0.25 + 0.03*traits["autonomic"]
    for t in range(STEPS):
        heart_phase += hr_base*DT + rng_global.normal(scale=0.01)
        resp_phase += rr_base*DT + rng_global.normal(scale=0.01)
        heart.append(np.sin(heart_phase))
        resp.append(np.sin(resp_phase))
    return np.array(heart), np.array(resp)

# ============================================================
# BRAIN DYNAMICS
# ============================================================

def simulate_brain(W, traits, heart):
    N = W.shape[0]
    z = rng_global.normal(size=N) + 1j*rng_global.normal(size=N)
    omega = rng_global.uniform(0.04,0.07,N)
    gain = 0.02 + 0.01*traits["neural_gain"]
    phases = np.zeros((STEPS,N))
    power = np.zeros((STEPS,N))
    row_sum = W.sum(axis=1)
    for t in range(STEPS):
        coupling = 0.6*(W@z - row_sum*z)
        dz = (gain + 1j*omega - np.abs(z)**2)*z + coupling
        dz += K_BODY*heart[t]
        z += dz*DT + 0.02*np.sqrt(DT)*(rng_global.normal(size=N) + 1j*rng_global.normal(size=N))
        phases[t] = np.angle(z)
        power[t] = np.abs(z)**2
    return phases, power

# ============================================================
# ENERGY DYNAMICS
# ============================================================

def simulate_energy(brain_power, heart, resp, traits):
    E = 1
    series = []
    metabolic = traits["metabolic"]
    decay = 0.02
    for t in range(STEPS):
        production = 0.4 + 0.2*heart[t] + 0.1*resp[t] + 0.1*metabolic
        consumption = 0.05*brain_power[t]
        dE = production - consumption - decay*E
        E += dE*DT
        series.append(E)
    return np.array(series)

# ============================================================
# METRICS
# ============================================================

def metastability(phases):
    R = np.abs(np.mean(np.exp(1j*phases), axis=1))
    return np.std(R)

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
# SINGLE SUBJECT SIMULATION WITH MULTIPLE MODELS
# ============================================================

def simulate_subject_models(W, seed_subj):
    traits = sample_traits(seed_subj)
    heart, resp = simulate_body(traits)
    phases, power = simulate_brain(W, traits, heart)
    brain_global = power.mean(axis=1)
    energy = simulate_energy(brain_global, heart, resp, traits)

    # Metrics
    m = metastability(phases)
    coh = brain_body_coherence(brain_global, heart)
    comp = predictive_complexity(power)
    stab, eff = energy_metrics(energy)
    cog = generate_cognition(traits)

    # Models
    # 1. Full model (Energy + Heart)
    vuln_full = rng_global.normal() - 0.5*eff + 0.5*stab

    # 2. Free-energy / Forward modeling only
    vuln_fe = rng_global.normal() - 0.5*comp

    # 3. Null model
    vuln_null = rng_global.normal()

    return {
        "Metastability": m,
        "BrainHeartCoherence": coh,
        "ForwardModeling": comp,
        "EnergyStability": stab,
        "EnergyEfficiency": eff,
        "Cognition": cog,
        "Vulnerability_Full": vuln_full,
        "Vulnerability_FE": vuln_fe,
        "Vulnerability_Null": vuln_null
    }

# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset():
    all_results = []
    for N in NETWORK_SIZES:
        for net_idx in range(N_NETWORKS_PER_SIZE):
            W = small_world_network(N, seed=net_idx)
            for seed_subj in range(SEEDS_PER_NETWORK):
                for subj in range(SUBJECTS_PER_SEED):
                    result = simulate_subject_models(W, seed_subj + subj)
                    result["NetworkSize"] = N
                    result["NetworkID"] = f"{N}_{net_idx}"
                    all_results.append(result)
    return pd.DataFrame(all_results)

# ============================================================
# CROSS-VALIDATION
# ============================================================

def crossval_r2(df, predictor, target):
    X = df[[predictor]].values
    y = df[target].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    kf = KFold(5, shuffle=True, random_state=SEED_GLOBAL)
    scores = []
    for tr, te in kf.split(X):
        model = LinearRegression()
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        scores.append(r2_score(y[te], pred))
    return np.mean(scores), np.std(scores)

# ============================================================
# PERMUTATION TEST
# ============================================================

def permutation_test(df, predictor, target, n_perm=100):
    real, _ = crossval_r2(df, predictor, target)
    null = []
    for _ in range(n_perm):
        perm = df.copy()
        perm[target] = rng_global.permutation(perm[target])
        r2, _ = crossval_r2(perm, predictor, target)
        null.append(r2)
    p = np.mean(np.array(null) >= real)
    return real, p

# ============================================================
# MEDIATION ANALYSIS
# ============================================================

def mediation_analysis(df, predictor, mediator, outcome):
    df_std = df[[predictor, mediator, outcome]].copy().apply(lambda x: (x - x.mean())/x.std())
    med_model = smf.ols(f"{mediator} ~ {predictor}", data=df_std).fit()
    out_model = smf.ols(f"{outcome} ~ {predictor} + {mediator}", data=df_std).fit()
    indirect = med_model.params[predictor] * out_model.params[mediator]
    direct = out_model.params[predictor]
    total = direct + indirect
    return {"Direct": direct, "Indirect": indirect, "Total": total}

# ============================================================
# RUN ROBUST SIMULATION
# ============================================================

print("Generating dataset with model comparison...")
df = generate_dataset()
print("Dataset size:", len(df))

# ============================================================
# SUMMARY TABLE
# ============================================================

summary = df.describe().T

# Cross-validation & permutation for all vulnerability models
for model in ["Vulnerability_Full","Vulnerability_FE","Vulnerability_Null"]:
    r2, p = permutation_test(df, "EnergyEfficiency", model)
    summary[f"CrossvalR2_{model}"] = r2
    summary[f"Perm_p_{model}"] = p

# Mediation: BrainHeartCoherence as mediator for each vulnerability
for model in ["Vulnerability_Full","Vulnerability_FE","Vulnerability_Null"]:
    med = mediation_analysis(df, "EnergyEfficiency", "BrainHeartCoherence", model)
    summary[f"Mediation_Direct_{model}"] = med["Direct"]
    summary[f"Mediation_Indirect_{model}"] = med["Indirect"]
    summary[f"Mediation_Total_{model}"] = med["Total"]

print("\nFINAL RESULTS TABLE WITH MODEL COMPARISON AND MEDIATION\n")
print(summary)

# ============================================================
# FIGURES
# ============================================================

# Scatter EnergyEfficiency vs Cognitive Vulnerability
plt.figure(figsize=(8,5))
sns.scatterplot(x="EnergyEfficiency", y="Vulnerability_Full", data=df, alpha=0.4, label="Full Model")
sns.scatterplot(x="EnergyEfficiency", y="Vulnerability_FE", data=df, alpha=0.4, label="Free-Energy")
sns.scatterplot(x="EnergyEfficiency", y="Vulnerability_Null", data=df, alpha=0.4, label="Null Model")
plt.title("Energy Efficiency vs Cognitive Vulnerability (All Models)")
plt.legend()
plt.show()

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Brain-Body-Energy Metrics Correlation Matrix")
plt.show()

# Mediation diagram example for Full model
med_full = mediation_analysis(df, "EnergyEfficiency", "BrainHeartCoherence", "Vulnerability_Full")
G = nx.DiGraph()
G.add_edge("EnergyEfficiency", "BrainHeartCoherence", label=f"{med_full['Direct']:.2f}")
G.add_edge("BrainHeartCoherence", "Vulnerability_Full", label=f"{med_full['Indirect']:.2f}")
G.add_edge("EnergyEfficiency", "Vulnerability_Full", label=f"{med_full['Total']:.2f}")
plt.figure(figsize=(6,4))
pos = {"EnergyEfficiency": (0,0), "BrainHeartCoherence": (1,0), "Vulnerability_Full": (2,0)}
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", arrowsize=20)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Mediation: EnergyEfficiency -> BrainHeartCoherence -> Vulnerability_Full")
plt.show()
