# CM-LLM: Cognitive Memory-Augmented Large Language Models

Reference implementation (skeleton) of the CM-LLM architecture — a cognitive memory system designed to augment LLMs with structured, time-aware, self-organising knowledge.

---

## What this is

Standard RAG systems re-derive knowledge from scratch on every query.  CM-LLM takes a different approach: it maintains a **living knowledge graph** that accumulates, decays, and self-organises over time, with a hard architectural guarantee that confabulated knowledge cannot enter the operational reasoning layer.

The five core components — distributional embeddings, claim-type decay, three-tier memory, structural bridges, and spectral domain clustering — are not five separate modules.  They are five views of one mechanism.

---

## Architecture overview

### 1. Distributional embeddings

Concepts are represented as Gaussian distributions N(μ, Σ) rather than point vectors.  Confidence is the inverse of the determinant of the covariance matrix — a geometric quantity, not a label.  Uncertainty is encoded in the geometry itself.

### 2. Claim-type-specific decay

Knowledge does not age uniformly by document date.  A claim extracted from a 2018 report decays at a rate determined by what the claim *is*, not when it was written:

| Claim type | Approximate half-life |
|---|---|
| `market_data` | ~1 week |
| `opinion` | ~3 months |
| `scientific_fact` | ~5 years |
| `mathematical` | ~50 years |

Decay is implemented as covariance expansion: without reinforcement, a concept's distribution diffuses — it becomes less precise, not simply older.

### 3. Three-tier memory

| Tier | Contents | Admission | Role |
|---|---|---|---|
| **Active** | Anchored, validated knowledge | stability > 0.9 **and** explicit anchor | Operational reasoning |
| **Episodic** | Plausible hypotheses | stability > 0.7 | Decay unless reinforced |
| **Cold storage** | Failed hypotheses, confabulation patterns | confidence < 0.1 | Calibration only — never queried for reasoning |

**Architectural invariant**: no node may enter active memory without a verified real-world anchor (source URL, document ID, direct observation).  This makes confabulation structurally impossible in the active layer — not probabilistically discouraged, but geometrically blocked.

### 4. Structural bridges (orthogonal Procrustes)

A bridge is a measurable orthogonal transformation between two memory regions.  If the same structure recurs in two distant domains, knowledge validated in one transfers to the other without re-validation.

Bridge quality is a scalar (normalised residual error), not a qualitative judgement.  Bridges compose: A→B and B→C imply a candidate bridge A→C.

Implementation: orthogonal Procrustes analysis — the same technique used for cross-lingual embedding alignment.

### 5. Spectral domain clustering

Domains are not predefined categories.  They emerge as clusters in the graph Laplacian spectrum.  The system discovers its own ontology and can track domain births, mergers, fissions, and drift over time.

---

## Architectural invariant

> No node can exist in active memory without an explicit real-world anchor.

Connections between anchored nodes inherit the confidence of their endpoints.  Unanchored hypotheses live only in episodic memory, where they decay naturally unless reinforced.  This resolves the proliferation problem (speculative knowledge overwhelming grounded knowledge) by construction, not by policy.

---

## Known limitations

**Stability criterion requires internal LLM access.**
`compute_stability` measures convergence of representations across transformer layers.  Most hosted APIs (OpenAI, Anthropic, etc.) do not expose intermediate layer embeddings.  For API-only deployments, substitute a fixed stability estimate or a token-level entropy proxy.

**Covariance initialisation is diagonal.**
Production systems should learn Σ from embedding variance across multiple source ingestions.

**The three-agent validation loop is not implemented.**
The full architecture includes an epistemic agent (generates hypotheses), a faithful amplifier (grounds them), and an adversarial falsifier (attempts to break them).  This skeleton implements only the memory substrate.

**Episodic → active promotion.**
`reinforce_node` implements the promotion path (episodic node gains an anchor and sufficient stability → moves to active).  The full reinforcement loop — detecting when an episodic node has accumulated enough evidence — is left for implementors.

**RecursiveMAS latent state sharing.**
The architecture theorises that heterogeneous agents should share latent states rather than text messages.  In practice, shared latent spaces across different model architectures require explicit alignment (e.g. a common embedding space).  This is not implemented here.

---

## Installation

```bash
pip install numpy scipy scikit-learn
```

Python 3.9+ required.

---

## Usage

```python
from cm_llm import CMLLMMemory
import numpy as np

memory = CMLLMMemory(embedding_dim=768)

# Add a node — layer_embeddings come from the LLM's intermediate layers
layer_embeddings = [np.random.randn(768) for _ in range(8)]  # replace with real embeddings
node_id, level = memory.add_node(
    anchor="https://example.com/source",
    claim_type="scientific_fact",
    layer_embeddings=layer_embeddings
)
print(f"Node admitted to: {level}")  # 'active', 'episodic', or 'cold'

# Apply daily decay
moved = memory.apply_decay_to_all(dt=1.0)

# Discover emergent domains
domains = memory.recluster_domains(n_clusters=5)

# Find a structural bridge between two domains
bridge = memory.find_bridge(domains[0], domains[1])
if bridge:
    print(f"Bridge quality (lower = better): {bridge.error:.3f}")

# Query active memory
query = np.random.randn(768)
results = memory.query(query, k=5)
```

Run the self-contained demo:

```bash
python cm_llm.py
```

---

## Claim types

```python
DECAY_COEFFICIENTS = {
    'market_data':     0.10,   # ~1 week half-life
    'opinion':         0.05,   # ~3 months
    'scientific_fact': 0.001,  # ~5 years
    'mathematical':    0.0001, # ~50 years
}
```

Add custom claim types by extending this dictionary.

---

## Theoretical background

The full theoretical specification is documented in `architettura_teorica_rivista.md`.  The key ideas are:

- **Decay as epistemic hygiene**: low-quality content receives no reinforcement from authoritative sources and self-eliminates; well-grounded knowledge survives.
- **Human grounding as load-bearing**: the validator regress problem (internal coherence ≠ truth) means periodic human validation is architecturally necessary, not an emergency override.
- **Geometry over ontology**: domain names and categories are interpretations of a continuous geometric structure.  The geometry is more stable than the ontology.

---

## File structure

```
cm_llm.py                       — reference implementation
architettura_teorica_rivista.md — full theoretical specification (revised edition)
architettura_secondo_cervello.md — original concept document
README.md                       — this file
```

---

## Roadmap

- [ ] Production embedding adapter (OpenAI / local Llama / Mistral)
- [ ] Entropy-based stability proxy for API-only deployments
- [ ] Three-agent validation loop (epistemic / faithful / adversarial)
- [ ] Persistent storage backend
- [ ] Diagonal → full covariance learning
- [ ] Bridge composition (transitive closure)

---

## Licence

MIT
