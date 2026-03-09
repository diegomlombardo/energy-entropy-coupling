# ============================================================
# ROBUST BRAIN–BODY–ENERGY IN SILICO MODEL
# Non-circular cognition, robust across seeds, networks, parameters
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

NETWORK_SIZES = [40, 80, 120]           # Different network sizes
N_NETWORKS_PER_SIZE = 3                 # Number of networks per size
SEEDS_PER_NETWORK = 3                   # Seeds for robustness
SUBJECTS_PER_SEED = 20                  # Subjects per seed
T = 200
DT = 0.05
STEPS = int(T / DT)
K_BODY_VALUES = [0.03, 0.04, 0.05]      # Test robustness to K_BODY
METABOLIC_GAIN_VALUES = [0.9, 1.0, 1.1] # Test metabolic parameter robustness

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
# ENERGY
# ============================================================

def simulate_energy(brain_power, heart, resp, traits, metabolic_gain=1.0):
    E = 1
    series = []
    decay = 0.02
    for t in range(STEPS):
        production = 0.4 + 0.2*heart[t] + 0.1*resp[t] + 0.1*traits["metabolic"]*metabolic_gain
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
# SUBJECT SIMULATION
# ============================================================

def simulate_subject(W, age, K_BODY=0.04, metabolic_gain=1.0):
    traits = sample_traits(seed=int(age*10))
    heart, resp = simulate_body(traits)
    phases, power = simulate_brain(W, traits, heart, K_BODY=K_BODY)
    brain_global = power.mean(axis=1)
    energy = simulate_energy(brain_global, heart, resp, traits, metabolic_gain=metabolic_gain)

    # Metrics
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
        "MetabolicGain": metabolic_gain
    }

# ============================================================
# GENERATE DATASET ROBUST ACROSS PARAMETERS, SEEDS, NETWORKS
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
                            res = simulate_subject(W, age, K_BODY=K_BODY, metabolic_gain=metabolic_gain)
                            res["NetworkSize"] = N
                            res["NetworkID"] = f"{N}_{net_idx}"
                            all_results.append(res)
    return pd.DataFrame(all_results)

# ============================================================
# CROSS-VALIDATION AND MEDIATION
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

def mediation_analysis(df, predictor, mediator, outcome):
    df_std = df[[predictor, mediator, outcome]].apply(lambda x: (x - x.mean())/x.std())
    med_model = smf.ols(f"{mediator} ~ {predictor}", data=df_std).fit()
    out_model = smf.ols(f"{outcome} ~ {predictor} + {mediator}", data=df_std).fit()
    indirect = med_model.params[predictor]*out_model.params[mediator]
    direct = out_model.params[predictor]
    total = direct + indirect
    return {"Direct": direct, "Indirect": indirect, "Total": total}

# ============================================================
# RUN FULL ROBUST SIMULATION
# ============================================================

print("Running robust simulation, this may take several hours...")
df = generate_dataset_robust()
print("Dataset complete:", len(df))

# ============================================================
# FOCUS ON KEY VARIABLES
# ============================================================

key_vars = ["BrainHeartCoherence","BrainRespCoherence","EnergyStability","EnergyEfficiency","Cognition"]
corr = df[key_vars].corr()
print("\nCORRELATION MATRIX (Key Variables)\n", corr)

# MEDIATION: EnergyEfficiency -> BrainBodyCoherence -> Cognition
med_hr = mediation_analysis(df, "EnergyEfficiency", "BrainHeartCoherence", "Cognition")
med_resp = mediation_analysis(df, "EnergyEfficiency", "BrainRespCoherence", "Cognition")
med_table = pd.DataFrame([med_hr, med_resp], index=["BrainHeartCoherence","BrainRespCoherence"])
print("\nMEDIATION RESULTS\n", med_table)

# CROSS-VALIDATED R² for key predictors
for var in ["BrainHeartCoherence","BrainRespCoherence","EnergyEfficiency","EnergyStability"]:
    r2, std = crossval_r2(df, var, "Cognition")
    print(f"{var} -> Cognition : R² = {r2:.3f} ± {std:.3f}")

# ============================================================
# LONGITUDINAL AGE TRENDS
# ============================================================

age_bins = np.linspace(20,80,13)
df["AgeBin"] = pd.cut(df["Age"], bins=age_bins)
trend = df.groupby("AgeBin").agg({
    "EnergyEfficiency":"mean",
    "BrainHeartCoherence":"mean",
    "BrainRespCoherence":"mean"
}).reset_index()

plt.figure(figsize=(8,5))
plt.plot(trend.index, trend["EnergyEfficiency"], label="EnergyEfficiency", marker='o')
plt.plot(trend.index, trend["BrainHeartCoherence"], label="BrainHeartCoherence", marker='x')
plt.plot(trend.index, trend["BrainRespCoherence"], label="BrainRespCoherence", marker='s')
plt.xticks(ticks=range(len(trend)), labels=[f"{int(interval.left)}-{int(interval.right)}" for interval in trend["AgeBin"]], rotation=45)
plt.ylabel("Mean Value")
plt.xlabel("Age Bin")
plt.title("Longitudinal Trends by Age")
plt.legend()
plt.tight_layout()
plt.show()
