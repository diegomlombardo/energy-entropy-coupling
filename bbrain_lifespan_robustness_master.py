import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, pearsonr
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import pickle
import os

# ============================================================
# GLOBAL PARAMETERS
# ============================================================
NETWORK_SIZES = [40, 80, 120]  # Multi-resolution networks
T = 200
DT = 0.05
STEPS = int(T / DT)
N_SEEDS = 50
SAVE_DIR = "BrainBodyModel_Results"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# CONNECTIVITY: SMALL-WORLD NETWORK
# ============================================================
def create_small_world(N=40, k=6, p=0.2):
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(1, k//2+1):
            W[i,(i+j)%N]=1
            W[i,(i-j)%N]=1
    # Rewiring
    for i in range(N):
        for j in range(N):
            if W[i,j]==1 and np.random.rand()<p:
                W[i,j]=0
                W[i,np.random.randint(N)]=1
    W = (W + W.T)/2
    np.fill_diagonal(W,0)
    eigvals = np.linalg.eigvals(W)
    W = W/np.max(np.abs(eigvals))
    return W

# ============================================================
# ECM MODEL
# ============================================================
def simulate_ECM(W,G=0.5,alpha=1.0,beta=0.6,noise=0.02):
    N = W.shape[0]
    z = np.random.randn(N)+1j*np.random.randn(N)
    omega = np.random.uniform(0.04,0.07,N)
    traj = np.zeros((STEPS,N))
    E = 1.0
    row = np.sum(W,axis=1)
    for t in range(STEPS):
        energy = np.mean(np.abs(z)**2)
        dE = alpha - beta*energy - 0.05*E
        E += dE*DT
        E = max(E,0.01)
        coupling = G*(W@z - row*z)
        dz = (0.02+1j*omega-np.abs(z)**2)*z + coupling - (1/E)*z
        dz += noise*(np.random.randn(N)+1j*np.random.randn(N))
        z += dz*DT
        traj[t] = np.abs(z)
    return traj

# ============================================================
# FEP MODEL
# ============================================================
def simulate_FEP(W,G=0.5,precision=1.0,noise=0.02):
    N = W.shape[0]
    z = np.random.randn(N)+1j*np.random.randn(N)
    omega = np.random.uniform(0.04,0.07,N)
    traj = np.zeros((STEPS,N))
    row = np.sum(W,axis=1)
    for t in range(STEPS):
        prediction = W@z
        pred_error = z - prediction
        coupling = G*(W@z - row*z)
        dz = (0.02+1j*omega-np.abs(z)**2)*z + coupling - precision*pred_error
        dz += noise*(np.random.randn(N)+1j*np.random.randn(N))
        z += dz*DT
        traj[t] = np.abs(z)
    return traj

# ============================================================
# NULL MODEL
# ============================================================
def simulate_null(N):
    return np.abs(np.random.randn(STEPS,N))

# ============================================================
# METRICS
# ============================================================
def compute_entropy_sync(traj):
    R = np.mean(traj,axis=1)
    hist = np.histogram(R,bins=40,density=True)[0]+1e-8
    return entropy(hist)

def compute_EECI(traj):
    H = compute_entropy_sync(traj)
    energy_var = np.var(traj)
    return H/(1+energy_var)

def compute_FECI(traj):
    H = compute_entropy_sync(traj)
    pred_error = np.diff(np.mean(traj,axis=1))
    return H/(1+np.var(pred_error))

def predictive_coding_metric(traj):
    R = np.mean(traj,axis=1)
    return np.mean(np.abs(np.diff(R)))

# ============================================================
# EMERGENT COGNITION
# ============================================================
def network_integration(traj):
    fc = np.corrcoef(traj.T)
    eigvals = np.linalg.eigvals(fc)
    return np.max(np.real(eigvals))

def emergent_cognition(traj):
    synchrony = np.mean(np.std(traj,axis=1))
    metastability = np.std(np.mean(traj,axis=1))
    integration = network_integration(traj)
    cog = (-0.4*synchrony + 0.7*metastability + 0.6*integration)
    return cog

# ============================================================
# DATASET GENERATION
# ============================================================
def generate_dataset(W,n_subjects=200):
    N = W.shape[0]
    rows = []
    for s in range(n_subjects):
        latent = np.random.normal()
        G = np.random.uniform(0.3,0.8)
        noise = np.random.uniform(0.01,0.05)
        traj_ecm = simulate_ECM(W,G,noise=noise)
        traj_fep = simulate_FEP(W,G,noise=noise)
        traj_null = simulate_null(N)
        EECI = compute_EECI(traj_ecm) + 0.05*latent
        FECI = compute_FECI(traj_fep) + 0.05*latent
        NullCI = compute_FECI(traj_null)
        PredCI = predictive_coding_metric(traj_fep) + 0.05*latent
        Cog = emergent_cognition(traj_fep) + np.random.normal(0,0.05)
        rows.append({
            "EECI":EECI,"FECI":FECI,"NullCI":NullCI,"PredCI":PredCI,
            "Cog":Cog,"Latent":latent
        })
    return pd.DataFrame(rows)

# ============================================================
# CROSS-VALIDATION & PERMUTATION
# ============================================================
def cross_validated_r2(df,predictor):
    X = df[[predictor]].values
    y = df["Cog"].values
    kf = KFold(n_splits=10,shuffle=True,random_state=0)
    scores = []
    for train,test in kf.split(X):
        m = LinearRegression()
        m.fit(X[train],y[train])
        pred = m.predict(X[test])
        r,_ = pearsonr(pred,y[test])
        scores.append(r**2)
    return np.mean(scores)

def permutation_test(df,predictor,n_perm=200):
    real = cross_validated_r2(df,predictor)
    perms=[]
    for _ in range(n_perm):
        df2=df.copy()
        df2[predictor] = np.random.permutation(df2[predictor])
        perms.append(cross_validated_r2(df2,predictor))
    p = np.mean(np.array(perms)>=real)
    return real,p

# ============================================================
# ROBUSTNESS: MULTI-RESOLUTION
# ============================================================
def run_robustness_multiresolution():
    all_results=[]
    for N_nodes in NETWORK_SIZES:
        print(f"\n=== Network size N={N_nodes} ===")
        for seed in range(N_SEEDS):
            np.random.seed(seed)
            W = create_small_world(N=N_nodes)
            df = generate_dataset(W)
            for m in ["EECI","FECI","NullCI","PredCI"]:
                r2,p = permutation_test(df,m)
                all_results.append({
                    "Seed":seed,"NetworkSize":N_nodes,"Model":m,"R2":r2,"p":p
                })
    res = pd.DataFrame(all_results)
    res.to_csv(SAVE_DIR+"/robustness_multiresolution.csv",index=False)
    return res

# ============================================================
# METRIC CORRELATION
# ============================================================
def metric_correlation(df):
    corr = df[["EECI","FECI","PredCI"]].corr()
    print("\nMetric Correlations:\n",corr)
    return corr

# ============================================================
# PLACEHOLDERS: PARTIAL REGRESSION & TEMPORAL INDEPENDENCE
# ============================================================
def partial_regression_placeholder():
    print("\n[INFO] Partial regression analysis not yet implemented.")

def temporal_independence_placeholder():
    print("\n[INFO] Temporal independence check not yet implemented.")

# ============================================================
# PUBLICATION-QUALITY FIGURES
# ============================================================
def plot_multiresolution_results(res):
    sns.set(style="whitegrid",context="talk")
    plt.figure(figsize=(12,6))
    sns.pointplot(data=res,x="Model",y="R2",hue="NetworkSize",ci=95,dodge=True)
    plt.title("Cross-validated R² across seeds and network sizes")
    plt.ylabel("R²")
    plt.savefig(SAVE_DIR+"/model_comparison_multiresolution.png",dpi=300)
    plt.show()

def summary_table(res):
    summary_rows=[]
    for model in ["EECI","FECI","NullCI","PredCI"]:
        for N_nodes in NETWORK_SIZES:
            sub = res[(res.Model==model) & (res.NetworkSize==N_nodes)]
            summary_rows.append({
                "Model":model,"NetworkSize":N_nodes,
                "R2_mean":sub.R2.mean(),"R2_std":sub.R2.std(),
                "p_mean":sub.p.mean(),"p_std":sub.p.std()
            })
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(SAVE_DIR+"/summary_table_multiresolution.csv",index=False)
    print(df_sum)
    return df_sum

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__=="__main__":
    print("=== FULL BRAIN-BODY GENERATIVE MODEL PIPELINE ===")
    results = run_robustness_multiresolution()
    plot_multiresolution_results(results)
    summary_table(results)
    # Additional analyses
    partial_regression_placeholder()
    temporal_independence_placeholder()
    # Metric correlation check
    df_last = generate_dataset(create_small_world(N=40))
    metric_correlation(df_last)
    print("\nPipeline complete. All results saved in:", SAVE_DIR)
