import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    n_subjects: int = 120
    timepoints: int = 20
    n_regions: int = 35
    dt: float = 0.05
    base_kappa: float = 0.5
    seeds: int = 50
    permutations: int = 500
    n_splits: int = 5
    noise_scale: float = 1.0
    kappa_sweep: tuple = (0.3,0.5,0.7)

CFG = Config()
np.seterr(all="raise")
WORLDS = ["energy","predictive","mixed","null"]

# ============================================================
# CONNECTOME / SUBJECT SIMULATION
# ============================================================

def spectral_normalize(W):
    eigvals = np.linalg.eigvals(W)
    rho = np.max(np.abs(eigvals))
    return W / (rho + 1e-8)

def generate_connectome(rng,n):
    W = rng.normal(0,1,(n,n))
    W = (W + W.T)/2
    return spectral_normalize(W)

def simulate_subject(world,rng,cfg,kappa_scale=1.0):
    W = generate_connectome(rng,cfg.n_regions)
    z = rng.normal(0,0.4,cfg.n_regions)
    rows=[]
    for t in range(cfg.timepoints):
        age = 20 + t*3
        kappa = cfg.base_kappa * kappa_scale
        dz = W @ z - kappa*z
        z = np.tanh(z + cfg.dt*dz)

        energy = np.mean(z**2)
        pred_error = np.var(W @ z - z)

        if world=="energy":
            cognition = np.exp(energy)
            pred_error = rng.normal(0,1)
        elif world=="predictive":
            cognition = np.tanh(pred_error)
            energy = rng.normal(0,1)
        elif world=="mixed":
            cognition = np.exp(energy) + np.tanh(pred_error)
        elif world=="null":
            cognition = rng.normal(0,1)
            energy = rng.normal(0,1)
            pred_error = rng.normal(0,1)

        cognition += rng.normal(0,cfg.noise_scale)
        rows.append([age,energy,pred_error,cognition])
        z += rng.normal(0,0.05,cfg.n_regions)
    return rows

def generate_dataset(world,seed,cfg,kappa_scale=1.0):
    rng = np.random.default_rng(seed)
    data=[]
    for s in range(cfg.n_subjects):
        subject_rows = simulate_subject(world,rng,cfg,kappa_scale)
        for row in subject_rows:
            data.append([s]+row)
    return pd.DataFrame(data,columns=["subject","age","energy","pred_error","cognition"])

# ============================================================
# CROSS-SECTIONAL AND LONGITUDINAL
# ============================================================

def cross_sectional_r2(df,cfg):
    X = df[["energy","pred_error"]].values
    y = df["cognition"].values
    groups = df["subject"].values
    X = StandardScaler().fit_transform(X)
    gkf = GroupKFold(n_splits=cfg.n_splits)
    r2_scores=[]
    betas=[]
    for tr,te in gkf.split(X,y,groups):
        model = Ridge(alpha=1.0)
        model.fit(X[tr],y[tr])
        pred = model.predict(X[te])
        r2_scores.append(r2_score(y[te],pred))
        betas.append(model.coef_)
    return np.mean(r2_scores), np.mean(betas,axis=0)

def longitudinal_vulnerability(df):
    slopes=[]
    early_energy=[]
    for s in df.subject.unique():
        sub = df[df.subject==s]
        model = LinearRegression()
        model.fit(sub[["age"]],sub["cognition"])
        slope = model.coef_[0]
        early_energy.append(sub[sub.age<40]["energy"].mean())
        slopes.append(-slope)
    dfv=pd.DataFrame({"early_energy":early_energy,"vulnerability":slopes})
    model = LinearRegression().fit(dfv[["early_energy"]],dfv["vulnerability"])
    return model.score(dfv[["early_energy"]],dfv["vulnerability"])

# ============================================================
# PERMUTATION TEST
# ============================================================

def permutation_test(df,true_r2,cfg,seed):
    rng = np.random.default_rng(seed)
    subjects = np.unique(df.subject.values)
    null_scores=[]
    for _ in range(cfg.permutations):
        shuffled = rng.permutation(subjects)
        mapping = dict(zip(subjects,shuffled))
        df_perm = df.copy()
        for orig,new in mapping.items():
            df_perm.loc[df.subject==orig,"cognition"]=df[df.subject==new]["cognition"].values
        r2,_ = cross_sectional_r2(df_perm,cfg)
        null_scores.append(r2)
    null_scores = np.array(null_scores)
    pval = (np.sum(null_scores>=true_r2)+1)/(len(null_scores)+1)
    return pval

# ============================================================
# IDENTIFY WORLD
# ============================================================

def identify_world(beta):
    bE,bP = beta
    if abs(bE)>abs(bP)*1.5: return "energy"
    if abs(bP)>abs(bE)*1.5: return "predictive"
    if abs(bE)>0.1 and abs(bP)>0.1: return "mixed"
    return "null"

# ============================================================
# EVALUATE WORLD
# ============================================================

def evaluate_world(world,cfg):
    cross_r2_list,long_r2_list,perm_sig_list,predicted_worlds=[],[],[],[]
    for seed in range(cfg.seeds):
        df=generate_dataset(world,seed,cfg)
        r2_cross,beta = cross_sectional_r2(df,cfg)
        r2_long = longitudinal_vulnerability(df)
        p = permutation_test(df,r2_cross,cfg,seed+999)
        predicted = identify_world(beta)
        cross_r2_list.append(r2_cross)
        long_r2_list.append(r2_long)
        perm_sig_list.append(p<0.05)
        predicted_worlds.append(predicted)
    return {
        "world":world,
        "cross_R2_mean":np.mean(cross_r2_list),
        "long_R2_mean":np.mean(long_r2_list),
        "perm_sig_rate":np.mean(perm_sig_list),
        "predicted_worlds":predicted_worlds,
        "cross_R2_se":np.std(cross_r2_list),
        "long_R2_se":np.std(long_r2_list)
    }

# ============================================================
# CROSS-WORLD MATRIX
# ============================================================

def cross_world_matrix(cfg):
    matrix={}
    for w1 in WORLDS:
        for w2 in WORLDS:
            r2s=[]
            for seed in range(cfg.seeds):
                df_train = generate_dataset(w1,seed,cfg)
                df_test = generate_dataset(w2,seed+999,cfg)
                X_train = StandardScaler().fit_transform(df_train[["energy","pred_error"]])
                y_train = df_train["cognition"].values
                X_test = StandardScaler().fit_transform(df_test[["energy","pred_error"]])
                y_test = df_test["cognition"].values
                model = Ridge(alpha=1.0).fit(X_train,y_train)
                pred = model.predict(X_test)
                r2s.append(r2_score(y_test,pred))
            matrix[(w1,w2)] = np.mean(r2s)
    return matrix

# ============================================================
# PARAMETER SWEEP
# ============================================================

def parameter_sweep(cfg):
    rows=[]
    for k in cfg.kappa_sweep:
        df = generate_dataset("energy",0,cfg,kappa_scale=k)
        rows.append({"kappa_scale":k,"mean_energy":df.energy.mean()})
    return pd.DataFrame(rows)

# ============================================================
# PLOT & SUMMARY
# ============================================================

def plot_summary(all_results,cw_matrix,param_df):
    # Table
    summary=[]
    for r in all_results:
        summary.append({
            "World":r["world"],
            "Cross-sectional R²":r["cross_R2_mean"],
            "Cross-sectional SE":r["cross_R2_se"],
            "Longitudinal R²":r["long_R2_mean"],
            "Longitudinal SE":r["long_R2_se"],
            "Permutation (%)":r["perm_sig_rate"]*100,
            "Recovery (%)":np.mean([r["world"]==p for p in r["predicted_worlds"]])*100
        })
    summary_df = pd.DataFrame(summary)
    print("\n=== SUMMARY TABLE ===\n")
    print(summary_df.to_string(index=False))

    # Figures style
    sns.set(style="whitegrid",context="talk",palette="Set2")

    # Cross-sectional
    plt.figure(figsize=(6,4))
    sns.barplot(x="World",y="Cross-sectional R²",data=summary_df,yerr=summary_df["Cross-sectional SE"])
    plt.title("Cross-sectional cognition prediction")
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig("cross_sectional_R2.png",dpi=300)
    plt.show()

    # Longitudinal vulnerability
    plt.figure(figsize=(6,4))
    sns.barplot(x="World",y="Longitudinal R²",data=summary_df,yerr=summary_df["Longitudinal SE"])
    plt.title("Longitudinal vulnerability (Energy → decline)")
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig("longitudinal_R2.png",dpi=300)
    plt.show()

    # Cross-world heatmap
    cw_df=pd.DataFrame([[cw_matrix[(w1,w2)] for w2 in WORLDS] for w1 in WORLDS],index=WORLDS,columns=WORLDS)
    plt.figure(figsize=(6,5))
    sns.heatmap(cw_df,annot=True,fmt=".2f",cmap="coolwarm",cbar_kws={"label":"R²"})
    plt.title("Cross-world generalization")
    plt.tight_layout()
    plt.savefig("cross_world_R2.png",dpi=300)
    plt.show()

    # Parameter sweep
    plt.figure(figsize=(6,4))
    sns.lineplot(x="kappa_scale",y="mean_energy",data=param_df,marker="o")
    plt.title("Parameter sweep: κ vs mean energy")
    plt.xlabel("Kappa scale")
    plt.ylabel("Mean energy")
    plt.tight_layout()
    plt.savefig("parameter_sweep.png",dpi=300)
    plt.show()

    return summary_df

# ============================================================
# MAIN
# ============================================================

if __name__=="__main__":
    all_results=[]
    for w in WORLDS:
        res = evaluate_world(w,CFG)
        all_results.append(res)

    cw = cross_world_matrix(CFG)
    param_df = parameter_sweep(CFG)

    summary_df = plot_summary(all_results,cw,param_df)

    # Proof-of-concept: energy predicts cognition vs predictive/null
    print("\n=== PROOF-OF-CONCEPT TEST: ENERGY vs OTHER WORLDS ===\n")
    for w in ["energy","predictive","null"]:
        df = generate_dataset(w,0,CFG)
        X = df[["energy"]].values
        y = df["cognition"].values
        r2 = LinearRegression().fit(X,y).score(X,y)
        print(f"{w.capitalize()} world: R² (energy → cognition) = {r2:.3f}")
