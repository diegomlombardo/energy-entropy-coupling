Brain–Body–Energy Generative Model

This repository contains a fully parallelized in-silico simulation framework designed to explore how brain–body–energy interactions shape cognition across lifespan-like cohorts. The model is inspired by systems-level theories suggesting that cognitive outcomes emerge not only from pathology, but from how metabolic stability and neural dynamics interact across scales. Here the pre-pint: http://dx.doi.org/10.2139/ssrn.6391438

Scientific Motivation

Traditional approaches to cognitive vulnerability often focus on molecular or pathological causes. Here, we take a different/complemetary perspective:

Cognition: We don’t yet have one complete mathematical model that fully explains how the human brain works. How we model it often depends on what we think the brain actually does. Making these models better is still an active and exciting area of research.

Large-scale brain dynamics are constrained by energy balance and autonomic signals.

Cognitive performance depends on the alignment between neural metastability and peripheral energy fluctuations, captured by what we call the Energy Coupling, under this view. 

By simulating synthetic cohorts across a wide age range, we can explore how coupling strength influences cognition, revealing patterns like the inverted-U trajectory, where optimal cognition emerges at intermediate coupling and declines at extremes.

In short, this framework allows researchers to test mechanistic hypotheses about energy, brain dynamics, and cognition, without relying on real-world pathology data.

What This Model Does

Simulates brain networks

Small-world networks with tunable size and connectivity.

Network dynamics modeled using graph Laplacians.

Models body oscillators

Cardiac and respiratory rhythms for each synthetic subject.

These peripheral signals interact bidirectionally with brain activity.

Simulates brain–energy dynamics

Neural activity modeled via Stuart-Landau oscillators.

Metabolic energy is dynamically produced and consumed based on brain and body signals.

Coupling parameters (K_BODY, K_ENERGY) control the influence of body and energy on the brain.

Computes metrics of system behavior

Metastability: how synchrony fluctuates across brain regions.

Brain–Heart Coherence: phase alignment between cardiac and neural signals.

Forward Modeling: predictive variability of neural power.

Energy Stability and Efficiency: variability and mean of metabolic energy.

Synthetic Cognition: linear combination of energy, body signals, and stochastic noise.

Fully parallelized

Uses all available CPU cores to simulate large cohorts quickly.

Supports multiple network sizes, random seeds, and hundreds of subjects per network.

Outputs

The simulation produces a comprehensive dataset of synthetic subjects:

BrainBodyEnergy_Publication/dataset_full.csv

Columns include: network ID, subject ID, age, coupling parameters, brain–body metrics, energy metrics, and synthetic cognition.

A NullModel column is included as a control.

Recommended Analyses

Partial correlations between predictors (BrainHeartCoherence, ForwardModeling, NullModel).

Robust regression predicting synthetic cognition.

Mediation analysis: EnergyEfficiency as a mediator between brain–body metrics and cognition.

Lifespan simulations: explore inverted-U trajectories of cognition versus coupling strength.

Running the Simulation
git clone <repository_url>
cd BrainBodyEnergy_Publication
python brain_body_energy_parallel.py

The script automatically detects your CPU cores and runs simulations in parallel.

Results are saved as .csv files for analysis.

Dependencies

Python ≥ 3.10

numpy

pandas

scipy

scikit-learn

statsmodels

joblib

matplotlib

seaborn

Notes

Fully in-silico — no human or animal data is required.

Designed for exploring mechanistic hypotheses linking energy, brain dynamics, and cognition.

Modular design allows easy extensions to new network topologies, coupling parameters, and cognitive metrics.

Brain–Body–Energy Generative Model

This repository contains a fully parallelized in-silico simulation framework designed to explore how brain–body–energy interactions shape cognition across lifespan-like cohorts. The model is inspired by systems-level theories suggesting that cognitive outcomes emerge not only from pathology, but from how metabolic stability and neural dynamics interact across scales.

Scientific Motivation

Traditional approaches to cognitive vulnerability often focus on molecular or pathological causes. Here, we take a different perspective:

Cognition is an emergent property of the whole brain–body system.

Large-scale brain dynamics are constrained by energy balance and autonomic signals.

Cognitive performance depends on the alignment between neural metastability and peripheral energy fluctuations.

By simulating synthetic cohorts across a wide age range, we can explore how coupling strength influences cognition, revealing patterns like the inverted-U trajectory, where optimal cognition emerges at intermediate coupling and declines at extremes.

In short, this framework allows researchers to test mechanistic hypotheses about energy, brain dynamics, and cognition, without relying on real-world pathology data.

What This Model Does

Simulates brain networks

Small-world networks with tunable size and connectivity.

Network dynamics modeled using graph Laplacians.

Models body oscillators

Cardiac and respiratory rhythms for each synthetic subject.

These peripheral signals interact bidirectionally with brain activity.

Simulates brain–energy dynamics

Neural activity modeled via Stuart-Landau oscillators.

Metabolic energy is dynamically produced and consumed based on brain and body signals.

Coupling parameters (K_BODY, K_ENERGY) control the influence of body and energy on the brain.

Computes metrics of system behavior

Metastability: how synchrony fluctuates across brain regions.

Brain–Heart Coherence: phase alignment between cardiac and neural signals.

Forward Modeling: predictive variability of neural power.

Energy Stability and Efficiency: variability and mean of metabolic energy.

Synthetic Cognition: linear combination of energy, body signals, and stochastic noise.

Fully parallelized

Uses all available CPU cores to simulate large cohorts quickly.

Supports multiple network sizes, random seeds, and hundreds of subjects per network.

Outputs

The simulation produces a comprehensive dataset of synthetic subjects:

BrainBodyEnergy_Publication/dataset_full.csv

Columns include: network ID, subject ID, age, coupling parameters, brain–body metrics, energy metrics, and synthetic cognition.

A NullModel column is included as a control.

Recommended Analyses

Partial correlations between predictors (BrainHeartCoherence, ForwardModeling, NullModel).

Robust regression predicting synthetic cognition.

Mediation analysis: EnergyEfficiency as a mediator between brain–body metrics and cognition.

Lifespan simulations: explore inverted-U trajectories of cognition versus coupling strength.

 Running the Simulation
git clone <repository_url>
cd BrainBodyEnergy_Publication
python brain_body_energy_parallel.py

The script automatically detects your CPU cores and runs simulations in parallel.

Results are saved as .csv files for analysis.

Dependencies

Python ≥ 3.10

numpy

pandas

scipy

scikit-learn

statsmodels

joblib

matplotlib

seaborn

Notes

Fully in-silico — no human or animal data is required.

Designed for exploring mechanistic hypotheses linking energy, brain dynamics, and cognition.

Modular design allows easy extensions to new network topologies, coupling parameters, and cognitive metrics.

References

Lombardo, D.M. (2026). Energy Coupling in Brain–Body Systems: An In-Silico Approach to Lifespan Cognitive and Computational Vulnerability.
