# Hierarchical Inverse Q-learning

Code accompanies the paper [Multi-intention Inverse Q-learning for Interpretable Behavior Representation](https://haozhu10015.github.io/papers/miiql.html).

## Abstract

In advancing the understanding of natural decision-making processes, inverse reinforcement learning (IRL) methods have proven instrumental in reconstructing animal's intentions underlying complex behaviors. Given the recent development of a continuous-time multi-intention IRL framework, there has been persistent inquiry into inferring discrete time-varying rewards with IRL. To address this challenge, we introduce the class of hierarchical inverse Q-learning (HIQL) algorithms. Through an unsupervised learning process, HIQL divides expert trajectories into multiple intention segments, and solves the IRL problem independently for each. Applying HIQL to simulated experiments and several real animal behavior datasets, our approach outperforms current benchmarks in behavior prediction and produces interpretable reward functions. Our results suggest that the intention transition dynamics underlying complex decision-making behavior is better modeled by a step function instead of a smoothly varying function. This advancement holds promise for neuroscience and cognitive science, contributing to a deeper understanding of decision-making and uncovering underlying brain mechanisms.

## Run the example

0. Install required packages.

```
pip install -r requirements.txt
```

1. Collect expert demonstrations.

```
python collect_demo.py
```

2. Train IAVI.

```
python train_iavi.py
```

3. Train HIAVI.

```
python train_hiavi.py
```

4. Plot learnt policy.

```
python plot.py
```
