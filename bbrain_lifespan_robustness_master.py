# ============================================================
# ROBUST BRAIN–BODY–ENERGY MODEL – NON-CIRCULAR COGNITION
# Mediation fixed: Brain-Body Coherence → Energy → Cognition
# ============================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# ============================================================
# GLOBAL PARAMETERS
# ============================================================

SEED_GLOBAL = 42
rng_global = np.random.default_rng(SEED_GLOBAL)

NETWORK_SIZES = [40, 80, 120]
N_NETWORKS_PER_SIZE = 3
SEEDS_PER_NETWORK = 3
SUBJECTS_PER_SEED = 20

T = 200
DT = 0.05
STEPS = int(T / DT)

K_BODY_VALUES = [0.03, 0.04, 0.05]
METABOLIC_GAIN_VALUES = [0.9, 1.0, 1.1]
NEURAL_GAIN_VALUES = [0.9, 1.0, 1.1]

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
    W = (W + W.T)/2
    np.fill_diagonal(W,0)
    eig = np.linalg.eigvals(W)
    W /= np.max(np.abs(eig))
    return W

def sample_traits(seed=None):
    rng = np.random.default_rng(seed)
    return {
        "autonomic": rng.normal(),
        "metabolic": rng.normal(),
        "neural_gain": rng.normal()
    }

# ============================================================
# BODY RHYTHMS
# ============================================================

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

def simulate_brain(W, traits, heart, K_BODY=0.04):
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
        z += dz*DT + 0.02*np.sqrt(DT)*(rng_global.normal(size=N)+1j*rng_global.normal(size=N))
        phases[t] = np.angle(z)
        power[t] = np.abs(z)**2
    return phases, power

# ============================================================
# ENERGY MODELS
# ============================================================

def simulate_energy(brain_power, heart, resp, traits, metabolic_gain=1.0, model="full"):
    E = 1
    series = []
    decay = 0.02
    for t in range(STEPS):
        if model == "full":
            production = 0.4 + 0.2*heart[t] + 0.1*resp[t] + 0.1*traits["metabolic"]*metabolic_gain
        elif model == "feedforward":
            production = 0.4 + 0.1*traits["metabolic"]*metabolic_gain
        elif model == "null":
            production = rng_global.normal(0.5,0.1)
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

def brain_body_coherence(brain, body):
    phase1 = np.angle(np.exp(1j*brain))
    phase2 = np.angle(np.exp(1j*body))
    return np.abs(np.mean(np.exp(1j*(phase1-phase2))))

def energy_metrics(E):
    return np.std(E), np.mean(E)

def generate_cognition(traits):
    return 0.5*traits["neural_gain"] + 0.4*traits["metabolic"] + 0.2*rng_global.normal()

# ============================================================
# SIMULATE SUBJECT
# ============================================================

def simulate_subject(W, age, K_BODY=0.04, metabolic_gain=1.0, energy_model="full"):
    traits = sample_traits(seed=int(age*10))
    heart, resp = simulate_body(traits)
    phases, power = simulate_brain(W, traits, heart, K_BODY=K_BODY)
    brain_global = power.mean(axis=1)
    energy = simulate_energy(brain_global, heart, resp, traits, metabolic_gain=metabolic_gain, model=energy_model)

    m = metastability(phases)
    coh_hr = brain_body_coherence(brain_global, heart)
    coh_resp = brain_body_coherence(brain_global, resp)
    stab, eff = energy_metrics(energy)
    cog = generate_cognition(traits)

    return {
        "Age": age,
        "Metastability": m,
        "BrainHeartCoherence": coh_hr,
        "BrainRespCoherence": coh_resp,
        "EnergyStability": stab,
        "EnergyEfficiency": eff,
        "Cognition": cog,
        "K_BODY": K_BODY,
        "MetabolicGain": metabolic_gain,
        "EnergyModel": energy_model
    }

# ============================================================
# ROBUST DATASET GENERATION
# ============================================================

def generate_dataset_robust():
    all_results = []
    for N in NETWORK_SIZES:
        for net_idx in range(N_NETWORKS_PER_SIZE):
            W = small_world_network(N, seed=net_idx)
            for seed_subj in range(SEEDS_PER_NETWORK):
                for subj in range(SUBJECTS_PER_SEED):
                    age = rng_global.uniform(20, 80)
                    for K_BODY in K_BODY_VALUES:
                        for metabolic_gain in METABOLIC_GAIN_VALUES:
                            for energy_model in ["full","feedforward","null"]:
                                res = simulate_subject(W, age, K_BODY=K_BODY, metabolic_gain=metabolic_gain, energy_model=energy_model)
                                res["NetworkSize"] = N
                                res["NetworkID"] = f"{N}_{net_idx}"
                                all_results.append(res)
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
        model = LinearRegression().fit(X[tr], y[tr])
        scores.append(r2_score(y[te], model.predict(X[te])))
    return np.mean(scores), np.std(scores)

# ============================================================
# MEDIATION: Brain-Body Coherence → Energy → Cognition
# ============================================================

def mediation_analysis(df, predictor, mediator, outcome):
    df_std = df[[predictor, mediator, outcome]].apply(lambda x: (x - x.mean())/x.std())
    med_model = smf.ols(f"{mediator} ~ {predictor}", data=df_std).fit()
    out_model = smf.ols(f"{outcome} ~ {predictor} + {mediator}", data=df_std).fit()
    indirect = med_model.params[predictor]*out_model.params[mediator]
    direct = out_model.params[predictor]
    total = direct + indirect
    return {"Direct": direct, "Indirect": indirect, "Total": total}

# ============================================================
# RUN SIMULATION
# ============================================================

print("Running robust simulation... this may take hours!")
df = generate_dataset_robust()
print("Simulation complete. Total subjects:", len(df))

# ============================================================
# CORRELATION MATRIX (KEY VARIABLES)
# ============================================================

key_vars = ["BrainHeartCoherence","BrainRespCoherence","EnergyEfficiency","EnergyStability","Cognition"]
corr = df[df["EnergyModel"]=="full"][key_vars].corr()
print("\nCorrelation matrix (full model):\n", corr)

# ============================================================
# MEDIATION ANALYSIS
# ============================================================

med_hr = mediation_analysis(df[df["EnergyModel"]=="full"], "BrainHeartCoherence", "EnergyEfficiency", "Cognition")
med_resp = mediation_analysis(df[df["EnergyModel"]=="full"], "BrainRespCoherence", "EnergyEfficiency", "Cognition")
med_table = pd.DataFrame([med_hr, med_resp], index=["BrainHeartCoherence","BrainRespCoherence"])
print("\nMediation results (Brain-Body → Energy → Cognition):\n", med_table)

# ============================================================
# CROSS-VALIDATED R²
# ============================================================

print("\nCross-validated R² (full model):")
for var in ["BrainHeartCoherence","BrainRespCoherence","EnergyEfficiency","EnergyStability"]:
    r2, std = crossval_r2(df[df["EnergyModel"]=="full"], var, "Cognition")
    print(f"{var} -> Cognition : R² = {r2:.3f} ± {std:.3f}")

# ============================================================
# LONGITUDINAL AGE TRENDS
# ============================================================

age_bins = np.linspace(20,80,13)
df["AgeBin"] = pd.cut(df["Age"], bins=age_bins)
trend = df.groupby("AgeBin")[["EnergyEfficiency","BrainHeartCoherence","BrainRespCoherence"]].mean().reset_index()

plt.figure(figsize=(8,5))
plt.plot(trend.index, trend["EnergyEfficiency"], label="EnergyEfficiency", marker='o')
plt.plot(trend.index, trend["BrainHeartCoherence"], label="BrainHeartCoherence", marker='x')
plt.plot(trend.index, trend["BrainRespCoherence"], label="BrainRespCoherence", marker='s')
plt.xticks(ticks=range(len(trend)), labels=[f"{int(interval.left)}-{int(interval.right)}" for interval in trend["AgeBin"]], rotation=45)
plt.ylabel("Mean Value")
plt.xlabel("Age Bin")
plt.title("Longitudinal Age Trends")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# VULNERABILITY SUMMARY TABLE
# ============================================================

summary = df.groupby("EnergyModel")[key_vars].agg(["mean","std"])
print("\nVulnerability summary table (mean ± SD by model):\n", summary)
