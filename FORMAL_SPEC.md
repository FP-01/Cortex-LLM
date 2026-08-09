# CM-LLM: Formal Specification

This document formalises the CM-LLM architecture as a dynamical system on a Hilbert space. It covers the five core components implemented in [`cm_llm.py`](./cm_llm.py): distributional embeddings, temporal decay, three-tier memory, structural bridges, and spectral domain clustering.

The presentation is self-contained. Familiarity with linear algebra and basic real analysis is assumed.

---

## Contents

1. [Mathematical Setup](#1-mathematical-setup)
2. [Dynamics](#2-dynamics)
3. [Stability](#3-stability)
4. [Memory Partitioning](#4-memory-partitioning)
5. [Structural Bridges](#5-structural-bridges)
6. [Retrieval](#6-retrieval)
7. [Convergence Results](#7-convergence-results)
8. [Loss Function](#8-loss-function)
9. [Summary](#9-summary-of-formal-properties)

---

## 1. Mathematical Setup

**Definition 1.1 (Embedding Space)**

Let $\mathcal{H} = \mathbb{R}^d$ be a finite-dimensional real Hilbert space with inner product

$$\langle x, y \rangle = \sum_{i=1}^{d} x_i y_i$$

and induced norm $\|x\| = \sqrt{\langle x, x \rangle}$.

---

**Definition 1.2 (Distributional Embedding)**

A concept is represented as a Gaussian measure $z \sim \mathcal{N}(\mu, \Sigma)$ where:

- $\mu \in \mathcal{H}$ is the mean embedding (semantic content)
- $\Sigma \in \mathbb{R}^{d \times d}$ is the covariance matrix (epistemic uncertainty), $\Sigma \succ 0$

Instantaneous confidence is defined as:

$$c_{\Sigma} = \frac{1}{\det(\Sigma + \varepsilon I)}$$

for a fixed regulariser $\varepsilon > 0$.

> **Design note.** $c_\Sigma$ is a geometric quantity derived from the distribution's spread: the smaller the volume of the uncertainty ellipsoid, the higher the confidence. It is distinct from the temporal confidence $c_i(t)$ defined in §2, which serves as the operational decay envelope. The two are reconciled in Remark 2.5.

---

**Definition 1.3 (Memory State)**

The memory at time $t$ is a finite set:

$$M(t) = \{(z_i, c_i, t_i)\}_{i=1}^{N(t)}$$

where $z_i \sim \mathcal{N}(\mu_i, \Sigma_i)$, $c_i \in (0, 1]$ is normalised confidence, and $t_i$ is the last update time.

The range of $c_i$ is the open-closed interval $(0, 1]$: because $\det(\Sigma + \varepsilon I) > 0$ for all finite $\Sigma$ and $\varepsilon > 0$, the value $c_i = 0$ is never attained in finite time. In practice $c_i$ is normalised by its value at initialisation.

---

## 2. Dynamics

**Definition 2.1 (Confidence Decay)**

The temporal evolution of confidence for concept $z_i$ is:

$$c_i(t) = c_i(0) \cdot e^{-\alpha_i t}, \qquad \alpha_i > 0$$

where $\alpha_i$ is the decay rate for the claim type of $z_i$ — not a function of document age.

Typical values:

| Claim type | $\alpha_i$ (per day) | Approximate half-life |
|---|---|---|
| `market_data` | 0.10 | ~1 week |
| `opinion` | 0.05 | ~3 months |
| `scientific_fact` | 0.001 | ~5 years |
| `mathematical` | 0.0001 | ~50 years |

---

**Definition 2.2 (Uncertainty Growth)**

Covariance evolves by isotropic diffusion:

$$\Sigma_i(t) = \Sigma_i(0) + \beta_i t \cdot I, \qquad \beta_i \geq 0$$

This is the geometric counterpart of Def. 2.1: as $\Sigma_i$ expands, the distribution spreads and the instantaneous confidence $c_\Sigma$ decreases.

---

**Remark 2.5 (Compatibility of the Two Decay Models)**

Definitions 2.1 and 2.2 describe decay at two levels of abstraction. If confidence is computed geometrically from Def. 2.2:

$$c_i(t) = \frac{1}{\det(\Sigma_i(0) + (\beta_i t + \varepsilon) I)}$$

this decays as $t^{-d}$ for large $t$ (polynomial), not exponentially. The exponential form in Def. 2.1 is a first-order approximation valid when $\beta_i t \ll \lambda_{\min}(\Sigma_i(0) + \varepsilon I)$.

**Operational resolution:** Def. 2.1 is the rule the system tracks (what `apply_decay` computes). Def. 2.2 is the geometric interpretation. For large $t$, the system uses the exponential rule and resets $\Sigma_i$ to be consistent with the current $c_i(t)$.

---

**Proposition 2.3 (Monotonic Decay)**

Confidence is monotonically decreasing:

$$\frac{d}{dt} c_i(t) = -\alpha_i \, c_i(t) < 0$$

*Proof.* Direct differentiation of $c_i(t) = c_i(0) e^{-\alpha_i t}$. $\blacksquare$

---

**Proposition 2.4 (Uncertainty Divergence)**

As $t \to \infty$, trace of covariance diverges:

$$\lim_{t \to \infty} \mathrm{Tr}(\Sigma_i(t)) = \infty$$

*Proof.* $\mathrm{Tr}(\Sigma_i(t)) = \mathrm{Tr}(\Sigma_i(0)) + d \cdot \beta_i t$, which is linear and unbounded in $t$ for $\beta_i > 0$. $\blacksquare$

---

## 3. Stability

**Definition 3.1 (Layer-wise Stability)**

Let $h^{(1)}, h^{(2)}, \ldots, h^{(L)}$ be the hidden states of a transformer across $L$ layers. Define the mean consecutive step size:

$$S = \frac{1}{L-1} \sum_{l=1}^{L-1} \|h^{(l+1)} - h^{(l)}\|$$

A small $S$ indicates that the representation has stabilised across depth — the concept is consistently encoded regardless of which layer is read.

---

**Definition 3.2 ($\varepsilon$-Stability)**

A sequence $\{h^{(l)}\}$ is $\varepsilon$-stable if every consecutive step is bounded:

$$\|h^{(l+1)} - h^{(l)}\| < \varepsilon \quad \text{for all } l = 1, \ldots, L-1$$

---

**Proposition 3.3 (Step-Summability Implies Cauchy)**

If $\sum_{l=1}^{\infty} \|h^{(l+1)} - h^{(l)}\| < \infty$, then $\{h^{(l)}\}$ is a Cauchy sequence.

*Proof.* For any $m > n$:

$$\|h^{(m)} - h^{(n)}\| \leq \sum_{l=n}^{m-1} \|h^{(l+1)} - h^{(l)}\| \leq \sum_{l=n}^{\infty} \|h^{(l+1)} - h^{(l)}\|$$

The right-hand side is the tail of a convergent series, so it tends to zero as $n \to \infty$, independently of $m > n$. This is precisely the Cauchy condition. $\blacksquare$

> **Note on $\varepsilon$-stability.** $\varepsilon$-stability (Def. 3.2) is necessary but not sufficient for the Cauchy property. The bound $\|h^{(m)} - h^{(n)}\| \leq (m-n)\varepsilon$ grows with $m-n$ and does not vanish as $m, n \to \infty$. Step-summability is the correct sufficient condition.

---

**Corollary 3.4 (Finite-Dimensional Convergence)**

In $\mathbb{R}^d$, every step-summable sequence converges.

*Proof.* $\mathbb{R}^d$ is complete; every Cauchy sequence converges. $\blacksquare$

---

## 4. Memory Partitioning

**Definition 4.1 (Three-Level Memory)**

Given thresholds $0 < \theta_E < \theta_A < 1$ and stability threshold $\varepsilon > 0$, partition $M(t)$ into:

$$M_A(t) = \{z_i \in M(t) : c_i > \theta_A \;\wedge\; S_i < \varepsilon \;\wedge\; \mathrm{anchor}(z_i) \neq \emptyset\}$$

$$M_E(t) = \{z_i \in M(t) : \theta_E < c_i \leq \theta_A\}$$

$$M_C(t) = \{z_i \in M(t) : c_i \leq \theta_E\}$$

where $\mathrm{anchor}(z_i)$ is a verified real-world source (URL, document ID, direct observation).

The three tiers and their roles:

| Tier | Name | Role |
|---|---|---|
| $M_A$ | Active memory | Operational reasoning; only anchored, stable, high-confidence nodes |
| $M_E$ | Episodic memory | Hypotheses under evaluation; decays unless reinforced |
| $M_C$ | Cold archive | Confabulation patterns and failed hypotheses; never queried for reasoning, used only for generator calibration |

**Architectural invariant.** The condition $\mathrm{anchor}(z_i) \neq \emptyset$ in the definition of $M_A$ is the formal statement of the core guarantee: no node can enter the operational reasoning layer without a verified real-world anchor. Confidence and stability are necessary but not sufficient. This makes confabulation structurally impossible in $M_A$ — not probabilistically discouraged, but geometrically blocked by the partition definition.

---

**Proposition 4.2 (Memory Flow)**

Concepts flow according to the following transitions:

$$M_E \xrightarrow{\;\mathrm{anchor} \;+\; c_i > \theta_A\;} M_A \xrightarrow{\;c_i \leq \theta_A\;} M_E \xrightarrow{\;c_i \leq \theta_E\;} M_C$$

The generic path under sustained decay is $M_A \to M_E \to M_C$. A node that decays from $M_A$ retains its anchor and re-enters $M_E$, where it continues to decay. Direct transition $M_A \to M_C$ occurs only when confidence crosses both thresholds within a single update step.

*Proof.* Follows from Definitions 2.1–2.2 and the threshold structure of Def. 4.1. $\blacksquare$

---

## 5. Structural Bridges

**Definition 5.1 (Bridge)**

Given two concept sets $X, Y \in \mathbb{R}^{n \times d}$ with corresponding rows, a bridge is the solution to:

$$R^* = \arg\min_{R \in O(d)} \|XR - Y\|_F$$

where $O(d)$ is the orthogonal group. $R^*$ is the optimal structure-preserving map from the embedding region of $X$ to that of $Y$.

---

**Proposition 5.2 (Existence and Form)**

The solution exists and is given by $R^* = VU^T$, where $X^T Y = U\Sigma V^T$ is the singular value decomposition.

*Proof.* Standard orthogonal Procrustes result (Fan and Hoffman, 1955). $\blacksquare$

---

**Definition 5.3 (Bridge Quality)**

$$q(X, Y) = \frac{\|XR^* - Y\|_F}{\max(\|X\|_F, \|Y\|_F)}$$

Lower $q$ indicates a better bridge. $q = 0$ if and only if $Y = XR$ for some $R \in O(d)$ (exact structural isomorphism). Normalisation by $\max(\|X\|_F, \|Y\|_F)$ makes $q$ symmetric: $q(X, Y) = q(Y, X)$ up to the choice of which region is rotated.

---

**Proposition 5.4 (Bridge Quality Bounds)**

$$0 \leq q(X, Y) \leq 2$$

*Proof.* Non-negativity is immediate from the definition. For the upper bound: since $R^*$ is orthogonal, $\|XR^*\|_F = \|X\|_F$. By the triangle inequality:

$$\|XR^* - Y\|_F \leq \|XR^*\|_F + \|Y\|_F = \|X\|_F + \|Y\|_F \leq 2\max(\|X\|_F, \|Y\|_F)$$

therefore $q \leq 2$. The bound is tight when $\|X\|_F = \|Y\|_F$ and $XR^*$ and $Y$ are antipodal. In practice, $q > 1$ signals that regions are structurally incompatible and no bridge should be created; the implementation uses $q < 0.3$ as the admission threshold. $\blacksquare$

---

**Corollary 5.5 (Bridge Composition)**

If $R^*_{AB}$ is a bridge from region $A$ to $B$ and $R^*_{BC}$ from $B$ to $C$, then $R^*_{AB} R^*_{BC}$ is a candidate bridge from $A$ to $C$ (also orthogonal, since $O(d)$ is closed under composition).

The quality of the composed bridge is bounded by $q(A, C) \leq q(A, B) + q(B, C) + O(q(A,B) \cdot q(B,C))$ for small individual errors.

---

## 6. Retrieval

**Definition 6.1 (Retrieval Score)**

For a query vector $q \in \mathcal{H}$, the retrieval score for concept $z_i \in M_A(t)$ is:

$$\mathrm{score}(z_i, q) = \lambda_1 \frac{\langle \mu_i, q \rangle}{\|\mu_i\| \|q\|} + \lambda_2 c_i - \lambda_3 \,\mathrm{Tr}(\Sigma_i)$$

with weights $\lambda_1, \lambda_2, \lambda_3 \geq 0$. The three terms balance semantic similarity (cosine), epistemic confidence, and distributional uncertainty respectively.

---

**Proposition 6.2 (Score Bounds)**

At any fixed time $t$:

$$-\lambda_1 - \lambda_3 \,\mathrm{Tr}(\Sigma_{\max}(t)) \leq \mathrm{score}(z_i, q) \leq \lambda_1 + \lambda_2$$

where $\Sigma_{\max}(t)$ is the covariance of the node with the largest trace in $M_A(t)$.

*Proof.* Cosine similarity $\in [-1, 1]$; confidence $c_i \in (0, 1]$; $\mathrm{Tr}(\Sigma_i) \geq 0$.

Upper bound achieved at cosine $= 1$, $c_i = 1$, $\mathrm{Tr}(\Sigma_i) = 0$, giving $\lambda_1 + \lambda_2$.

Lower bound achieved at cosine $= -1$, $c_i \to 0$, $\mathrm{Tr}(\Sigma_i) = \mathrm{Tr}(\Sigma_{\max}(t))$, giving $-\lambda_1 - \lambda_3\,\mathrm{Tr}(\Sigma_{\max}(t))$.

Since $M_A(t)$ is a finite set at every finite $t$, the lower bound is finite at every query time. The infimum over all $t \to \infty$ is $-\infty$ (by Prop. 2.4, $\mathrm{Tr}(\Sigma_i)$ is unbounded) but this limit is never reached in a running system that moves decayed nodes to $M_C$. $\blacksquare$

---

## 7. Convergence Results

**Theorem 7.1 (Active Memory Stabilisation)**

Under sustained reinforcement — i.e., for each $z_i \in M_A$, a periodic reinforcement step resets $c_i$ to above $\theta_A$ and pulls $\Sigma_i$ back toward $\Sigma_i(0)$ — the parameters $(\mu_i, \Sigma_i)$ of each active node converge to a fixed point $(\mu_i^*, \Sigma_i^*)$.

*Proof sketch.* The update map $F: (\mu_i, \Sigma_i) \mapsto (\mu_i', \Sigma_i')$ combining one decay step (Def. 2.1–2.2) with one reinforcement step (a pull toward the anchored value) is a contraction on the parameter space $\mathbb{R}^d \times \mathbb{S}^d_{++}$ under the product metric (Euclidean on $\mathbb{R}^d$, Frobenius on $\mathbb{S}^d_{++}$). By the Banach fixed-point theorem, the sequence $F^n(\mu_i(0), \Sigma_i(0))$ converges to the unique fixed point. $\blacksquare$

---

**Theorem 7.2 (Episodic Decay)**

Without reinforcement, every node in $M_E(t)$ eventually transitions to $M_C$:

$$\lim_{t \to \infty} M_E(t) = \emptyset$$

*Proof.* By Prop. 2.3, $c_i(t) = c_i(0) e^{-\alpha_i t} \to 0$ as $t \to \infty$. Since $\alpha_i > 0$, there exists a finite time $T_i = \alpha_i^{-1} \log(c_i(0) / \theta_E)$ at which $c_i(T_i) = \theta_E$, so $z_i$ transitions to $M_C$. Since $M_E(t)$ is finite, $M_E(t) = \emptyset$ for all $t \geq \max_i T_i$. $\blacksquare$

---

**Theorem 7.3 (Bridge Stability)**

**(i)** If two regions are exactly isomorphic, i.e., $Y = XR_0$ for some $R_0 \in O(d)$, then $q(X, Y) = 0$.

**(ii)** If $Y = XR_0 + E$ where $\|E\|_F \leq \delta \cdot \max(\|X\|_F, \|Y\|_F)$, then:

$$q(X, Y) \leq \delta + O(\delta^2)$$

*Proof.*

**(i)** When $E = 0$: Prop. 5.2 gives $R^* = R_0$ as the SVD solution, and $\|XR_0 - Y\|_F = 0$, so $q = 0$.

**(ii)** When $E \neq 0$: by perturbation theory for the SVD (Wedin, 1972), $R^*$ deviates from $R_0$ by $O(\delta)$ in Frobenius norm. The residual satisfies:

$$\|XR^* - Y\|_F \leq \|E\|_F + O(\delta^2 \|X\|_F) \leq \delta \cdot \max(\|X\|_F, \|Y\|_F) + O(\delta^2 \|X\|_F)$$

Dividing by $\max(\|X\|_F, \|Y\|_F)$ gives $q \leq \delta + O(\delta^2)$. $\blacksquare$

---

## 8. Loss Function

**Definition 8.1 (Epistemic Regularisation)**

The training loss augments a standard language modelling objective with two regularisation terms:

$$\mathcal{L} = \mathcal{L}_{\mathrm{LM}} + \lambda_1 \mathcal{L}_{\mathrm{stab}} + \lambda_2 \mathcal{L}_{\mathrm{coh}}$$

**Stability term** — penalises representation drift across transformer layers:

$$\mathcal{L}_{\mathrm{stab}} = \sum_{l=1}^{L-1} \|h^{(l+1)} - h^{(l)}\|$$

Minimising $\mathcal{L}_{\mathrm{stab}}$ encourages step-summable (and therefore convergent) layer sequences, directly linking the training objective to Prop. 3.3.

**Coherence term** — enforces proximity of concepts declared consistent by the validation step:

$$\mathcal{L}_{\mathrm{coh}} = \sum_{(i,j) \in \mathcal{C}} \max\!\left(0,\, \delta - \frac{\langle \mu_i, \mu_j \rangle}{\|\mu_i\| \|\mu_j\|}\right)$$

where $\mathcal{C}$ is the set of pairs declared consistent by the validation step. Similarity is measured as cosine similarity of mean embeddings, consistent with the retrieval score (Def. 6.1).

> **Alternative.** If full distributional similarity is required, the symmetric KL divergence between $\mathcal{N}(\mu_i, \Sigma_i)$ and $\mathcal{N}(\mu_j, \Sigma_j)$ has a closed form and can replace the cosine term:
>
> $$D_{\mathrm{KL}}^{\mathrm{sym}}(z_i, z_j) = \frac{1}{2}\left[\mathrm{Tr}(\Sigma_j^{-1}\Sigma_i + \Sigma_i^{-1}\Sigma_j) + (\mu_i - \mu_j)^T(\Sigma_i^{-1} + \Sigma_j^{-1})(\mu_i - \mu_j) - 2d\right]$$

---

**Proposition 8.2 (Loss Non-negativity)**

$\mathcal{L} \geq \mathcal{L}_{\mathrm{LM}} \geq 0$.

*Proof.* $\mathcal{L}_{\mathrm{stab}} \geq 0$ (sum of norms). $\mathcal{L}_{\mathrm{coh}} \geq 0$ (hinge loss). $\blacksquare$

---

## 9. Summary of Formal Properties

| Property | Formal statement | Reference |
|---|---|---|
| Confidence decay | $c_i(t) = c_i(0) e^{-\alpha_i t}$, monotonically decreasing | Prop. 2.3 |
| Uncertainty growth | $\mathrm{Tr}(\Sigma_i(t))$ grows linearly; geometric confidence decays as $t^{-d}$ | Prop. 2.4, Rem. 2.5 |
| Convergence condition | Step-summability ($\sum \|h^{(l+1)} - h^{(l)}\| < \infty$) implies Cauchy; $\varepsilon$-stability alone is not sufficient | Prop. 3.3, Cor. 3.4 |
| Memory flow | $M_E \to M_A$ (reinforcement + anchor), $M_A \to M_E \to M_C$ (decay) | Prop. 4.2 |
| Anchor invariant | $M_A$ requires verified anchor; confabulation structurally excluded | Def. 4.1 |
| Bridge existence | Optimal $R^* \in O(d)$ exists, given by SVD (Procrustes) | Prop. 5.2 |
| Bridge quality | $q \in [0, 2]$, symmetric in $X$ and $Y$ | Prop. 5.4 |
| Bridge composition | $O(d)$ closed under composition; composed bridge quality bounded by sum of individual errors | Cor. 5.5 |
| Score bounds | $\mathrm{score} \leq \lambda_1 + \lambda_2$; bounded below at each finite $t$ | Prop. 6.2 |
| Active memory convergence | Parameters $(\mu_i, \Sigma_i)$ converge under reinforcement (Banach contraction on $\mathbb{R}^d \times \mathbb{S}^d_{++}$) | Thm. 7.1 |
| Episodic memory decay | $M_E(t) = \emptyset$ for $t \geq \max_i T_i$ without reinforcement | Thm. 7.2 |
| Bridge stability | $q = 0$ for exact isomorphism; $q \leq \delta + O(\delta^2)$ for $\delta$-perturbations | Thm. 7.3 |

---

## References

- Fan, K., & Hoffman, A. J. (1955). Some metric inequalities in the space of matrices. *Proceedings of the American Mathematical Society*, 6(1), 111–116. *(Orthogonal Procrustes)*
- Wedin, P.-Å. (1972). Perturbation bounds in connection with singular value decomposition. *BIT Numerical Mathematics*, 12(1), 99–111. *(SVD perturbation theory)*
- Rudin, W. (1976). *Principles of Mathematical Analysis* (3rd ed.). McGraw-Hill. *(Cauchy sequences, completeness)*
