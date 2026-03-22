"""
Layer Shuffle Recovery Experiment
=================================
Shuffle all decoder layers in a transformer (Qwen3-1.7B), then investigate
methods to recover the correct layer order using:
  - Weight-only (math) methods (no data needed)
  - Dataset-based methods (using vLLM rollouts on GSM8K as calibration)

Each method produces a recovered permutation, evaluated against ground truth
via Kendall tau correlation and positional accuracy.
"""

import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as scipy_stats
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import laplacian as graph_laplacian
from transformers import AutoModelForCausalLM
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "Qwen/Qwen3-1.7B"
SEED = 42
DEVICE = "cuda:0"
RESULTS_DIR = Path(__file__).parent / "results"
CALIB_NUM_SAMPLES = 64
CALIB_MAX_LEN = 256
SA_ITERATIONS = 1500
SA_T_INIT = 2.0

# ============================================================================
# Utilities
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def kendall_tau(recovered, ground_truth):
    tau, _ = scipy_stats.kendalltau(recovered, ground_truth)
    return tau


def positional_accuracy(recovered, ground_truth):
    return float((np.array(recovered) == np.array(ground_truth)).mean())


def displacement_score(recovered, ground_truth):
    return float(np.abs(np.array(recovered) - np.array(ground_truth)).mean())


def evaluate(recovered, num_layers):
    ideal = list(range(num_layers))
    return {
        "kendall_tau": kendall_tau(recovered, ideal),
        "accuracy": positional_accuracy(recovered, ideal),
        "displacement": displacement_score(recovered, ideal),
    }


# ============================================================================
# Model Loading & Layer Shuffling
# ============================================================================

def load_model():
    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    model.eval()
    num_layers = len(model.model.layers)
    print(f"Model loaded: {num_layers} decoder layers on {DEVICE}")
    return model, num_layers


def shuffle_layers(model, num_layers):
    """Shuffle decoder layers. Returns perm where perm[shuffled_pos] = original_layer_idx."""
    gen = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(num_layers, generator=gen).numpy()

    original_layers = list(model.model.layers)
    model.model.layers = nn.ModuleList([original_layers[perm[i]] for i in range(num_layers)])

    print(f"Layers shuffled. Ground truth perm: {perm.tolist()}")
    return perm


# ============================================================================
# Weight Feature Extraction (cached, shared across math methods)
# ============================================================================

class WeightFeatures:
    """Precompute and cache all weight-derived features once."""

    def __init__(self, model, num_layers):
        print("\nExtracting weight features...")
        t0 = time.time()
        self.num_layers = num_layers

        # Flatten all weights per layer -> list of 1D tensors
        self.flat_vectors = []
        self.norm_vectors = []
        self.stats = []
        self.svd_spectra = []

        for i in range(num_layers):
            layer = model.model.layers[i]

            # Full flattened weights
            parts = [p.data.float().cpu().flatten() for p in layer.parameters()]
            flat = torch.cat(parts)
            self.flat_vectors.append(flat)

            # Norm-only weights
            norm_parts = []
            for name, p in layer.named_parameters():
                if 'norm' in name.lower() and 'weight' in name.lower():
                    norm_parts.append(p.data.float().cpu().flatten())
            self.norm_vectors.append(torch.cat(norm_parts) if norm_parts else torch.zeros(1))

            # Scalar statistics
            self.stats.append(np.array([
                flat.norm().item(),
                flat.mean().item(),
                flat.std().item(),
                flat.abs().max().item(),
            ]))

            # SVD spectra from multiple weight matrices on GPU
            svd_parts = []
            for wname in ['self_attn.q_proj', 'self_attn.k_proj',
                          'self_attn.v_proj', 'mlp.gate_proj']:
                # Navigate dotted name
                obj = layer
                for attr in wname.split('.'):
                    obj = getattr(obj, attr)
                w = obj.weight.data.float()
                s = torch.linalg.svdvals(w)[:32]
                svd_parts.append(s.cpu().numpy())
            self.svd_spectra.append(np.concatenate(svd_parts))

        self.stats = np.array(self.stats)
        self.svd_spectra = np.array(self.svd_spectra)

        # Precompute pairwise distance matrices using scipy pdist (vectorized)
        flat_matrix = torch.stack(self.flat_vectors).numpy()
        norm_matrix = torch.stack(self.norm_vectors).numpy()

        self.D_cosine_full = squareform(pdist(flat_matrix, metric='cosine'))
        self.D_l2_norm = squareform(pdist(norm_matrix, metric='euclidean'))
        self.D_l2_stats = squareform(pdist(self.stats, metric='euclidean'))
        self.D_l2_svd = squareform(pdist(self.svd_spectra, metric='euclidean'))

        elapsed = time.time() - t0
        print(f"  Weight features extracted in {elapsed:.1f}s")


# ============================================================================
# Chain Building Heuristics
# ============================================================================

def greedy_nearest_neighbor_chain(D):
    """Build a Hamiltonian path via nearest-neighbor heuristic, trying all starts."""
    n = D.shape[0]
    best_path, best_cost = None, float('inf')
    for s in range(n):
        visited = {s}
        path = [s]
        for _ in range(n - 1):
            cur = path[-1]
            dists = D[cur].copy()
            dists[list(visited)] = float('inf')
            nxt = int(np.argmin(dists))
            path.append(nxt)
            visited.add(nxt)
        cost = sum(D[path[i], path[i+1]] for i in range(n-1))
        if cost < best_cost:
            best_path, best_cost = path, cost
    return best_path


def two_opt_improve(D, path):
    """2-opt local search for open-path TSP."""
    n = len(path)
    path = list(path)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                old_cost = D[path[i], path[i+1]]
                new_cost = D[path[i], path[j]]
                if j < n - 1:
                    old_cost += D[path[j], path[j+1]]
                    new_cost += D[path[i+1], path[j+1]]
                if new_cost < old_cost - 1e-10:
                    path[i+1:j+1] = reversed(path[i+1:j+1])
                    improved = True
    return path


def orient_chain(chain, model):
    """Determine chain direction using correlation of weight norms with position.

    Computes total weight norm for each layer along the chain, then checks
    whether the norms correlate positively with position (they tend to grow
    with depth in trained transformers). Uses Spearman rank correlation over
    the full chain — much more robust than comparing only the two endpoints.
    """
    n = len(chain)
    norms = []
    for idx in chain:
        layer = model.model.layers[idx]
        total = sum(p.data.float().norm().item() ** 2 for p in layer.parameters()) ** 0.5
        norms.append(total)

    positions = np.arange(n)
    corr_fwd = scipy_stats.spearmanr(positions, norms).statistic
    # If positive correlation with position, chain direction is correct
    if corr_fwd >= 0:
        return chain
    return list(reversed(chain))


def chain_to_permutation(chain, ground_truth_perm):
    """Convert chain of shuffled indices to recovered original-layer ordering.

    chain[i] = shuffled index at position i
    ground_truth_perm[s] = original layer at shuffled position s
    Result: recovered[i] = original layer that should be at position i
    """
    return [int(ground_truth_perm[chain[i]]) for i in range(len(chain))]


# ============================================================================
# Math-Only Methods (no dataset)
# ============================================================================

def method_weight_stats_continuity(model, num_layers, ground_truth_perm, wf: WeightFeatures):
    """Recover order by minimizing total variation of weight statistics."""
    print("\n[Method 1] Weight Statistics Continuity")
    t0 = time.time()

    chain = greedy_nearest_neighbor_chain(wf.D_l2_stats)
    chain = two_opt_improve(wf.D_l2_stats, chain)
    chain = orient_chain(chain, model)
    recovered = chain_to_permutation(chain, ground_truth_perm)

    elapsed = time.time() - t0
    metrics = evaluate(recovered, num_layers)
    print(f"  Kendall tau: {metrics['kendall_tau']:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
          f"Displacement: {metrics['displacement']:.2f}, Time: {elapsed:.1f}s")
    return {"method": "weight_stats_continuity", "type": "math",
            "recovered": recovered, "time": elapsed, **metrics}


def method_svd_spectrum(model, num_layers, ground_truth_perm, wf: WeightFeatures):
    """Recover order by SVD spectrum similarity of q_proj weights."""
    print("\n[Method 2] SVD Spectrum Similarity")
    t0 = time.time()

    chain = greedy_nearest_neighbor_chain(wf.D_l2_svd)
    chain = two_opt_improve(wf.D_l2_svd, chain)
    chain = orient_chain(chain, model)
    recovered = chain_to_permutation(chain, ground_truth_perm)

    elapsed = time.time() - t0
    metrics = evaluate(recovered, num_layers)
    print(f"  Kendall tau: {metrics['kendall_tau']:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
          f"Displacement: {metrics['displacement']:.2f}, Time: {elapsed:.1f}s")
    return {"method": "svd_spectrum", "type": "math",
            "recovered": recovered, "time": elapsed, **metrics}


def method_layernorm_progression(model, num_layers, ground_truth_perm, wf: WeightFeatures):
    """Recover order using RMSNorm weight progression."""
    print("\n[Method 3] LayerNorm Progression")
    t0 = time.time()

    chain = greedy_nearest_neighbor_chain(wf.D_l2_norm)
    chain = two_opt_improve(wf.D_l2_norm, chain)
    chain = orient_chain(chain, model)
    recovered = chain_to_permutation(chain, ground_truth_perm)

    elapsed = time.time() - t0
    metrics = evaluate(recovered, num_layers)
    print(f"  Kendall tau: {metrics['kendall_tau']:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
          f"Displacement: {metrics['displacement']:.2f}, Time: {elapsed:.1f}s")
    return {"method": "layernorm_progression", "type": "math",
            "recovered": recovered, "time": elapsed, **metrics}


def method_tsp_full_weights(model, num_layers, ground_truth_perm, wf: WeightFeatures):
    """Recover order via TSP on full weight cosine distance matrix."""
    print("\n[Method 4] TSP on Full Weight Distance")
    t0 = time.time()

    chain = greedy_nearest_neighbor_chain(wf.D_cosine_full)
    chain = two_opt_improve(wf.D_cosine_full, chain)
    chain = orient_chain(chain, model)
    recovered = chain_to_permutation(chain, ground_truth_perm)

    elapsed = time.time() - t0
    metrics = evaluate(recovered, num_layers)
    print(f"  Kendall tau: {metrics['kendall_tau']:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
          f"Displacement: {metrics['displacement']:.2f}, Time: {elapsed:.1f}s")
    return {"method": "tsp_full_weights", "type": "math",
            "recovered": recovered, "time": elapsed, **metrics}


def method_greedy_cosine(model, num_layers, ground_truth_perm, wf: WeightFeatures):
    """Greedy chain via cosine similarity, starting from most isolated layer."""
    print("\n[Method 5] Greedy Cosine Chain")
    t0 = time.time()

    S = 1.0 - wf.D_cosine_full  # similarity matrix
    avg_sim = S.sum(axis=1) / (num_layers - 1)
    start = int(np.argmin(avg_sim))

    visited = {start}
    chain = [start]
    for _ in range(num_layers - 1):
        cur = chain[-1]
        sims = S[cur].copy()
        sims[list(visited)] = -float('inf')
        nxt = int(np.argmax(sims))
        chain.append(nxt)
        visited.add(nxt)

    chain = orient_chain(chain, model)
    recovered = chain_to_permutation(chain, ground_truth_perm)

    elapsed = time.time() - t0
    metrics = evaluate(recovered, num_layers)
    print(f"  Kendall tau: {metrics['kendall_tau']:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
          f"Displacement: {metrics['displacement']:.2f}, Time: {elapsed:.1f}s")
    return {"method": "greedy_cosine", "type": "math",
            "recovered": recovered, "time": elapsed, **metrics}


def method_activation_flow(model, num_layers, ground_truth_perm):
    """Recover order by analyzing how each layer transforms random activations.

    Uses high-dimensional fingerprints for maximum discrimination:
    - Full channel-wise residual vectors (2048-d) instead of scalar stats
    - Multiple random probes at different scales for robustness
    - Concatenated into a rich per-layer feature vector

    Early layers (0-16) have nearly identical weight norms (~250), so scalar
    stats can't tell them apart. Channel-wise vectors capture which specific
    hidden dimensions each layer modifies, which IS unique per layer.
    """
    print("\n[Method 6] Activation Flow Analysis")
    t0 = time.time()

    hidden_size = model.config.hidden_size
    n_probes = 4
    fingerprints = []

    with torch.no_grad():
        for i in range(num_layers):
            layer = model.model.layers[i]
            parts = []

            for probe_idx in range(n_probes):
                torch.manual_seed(SEED + probe_idx)
                # Different scales to probe different activation regimes
                scale = [0.05, 0.1, 0.5, 1.0][probe_idx]
                fake_hidden = torch.randn(2, 32, hidden_size,
                                          dtype=torch.bfloat16, device=DEVICE) * scale
                pos_ids = torch.arange(32, device=DEVICE).unsqueeze(0).expand(2, -1)
                position_embeddings = model.model.rotary_emb(fake_hidden, pos_ids)

                output = layer(fake_hidden, position_ids=pos_ids,
                              position_embeddings=position_embeddings, use_cache=False)
                out = output[0] if isinstance(output, tuple) else output
                residual = (out - fake_hidden).float()

                # Full channel-wise fingerprint (2048-d per probe)
                channel_means = residual.mean(dim=(0, 1)).cpu().numpy()  # (hidden_size,)
                channel_stds = residual.std(dim=(0, 1)).cpu().numpy()
                parts.append(channel_means)
                parts.append(channel_stds)

            fingerprints.append(np.concatenate(parts))

    fingerprints = np.array(fingerprints)
    print(f"  Fingerprint shape: {fingerprints.shape} per layer")
    D = squareform(pdist(fingerprints, metric='euclidean'))
    chain = greedy_nearest_neighbor_chain(D)
    chain = two_opt_improve(D, chain)
    chain = orient_chain(chain, model)
    recovered = chain_to_permutation(chain, ground_truth_perm)

    elapsed = time.time() - t0
    metrics = evaluate(recovered, num_layers)
    print(f"  Kendall tau: {metrics['kendall_tau']:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
          f"Displacement: {metrics['displacement']:.2f}, Time: {elapsed:.1f}s")
    return {"method": "activation_flow", "type": "math",
            "recovered": recovered, "time": elapsed, **metrics}


def method_fiedler_spectral(model, num_layers, ground_truth_perm, wf: WeightFeatures):
    """Recover order via spectral ordering (Fiedler vector of similarity graph).

    The Fiedler vector (2nd smallest eigenvector of the graph Laplacian) gives
    a global ordering that minimizes a balanced cut objective. Sorting by Fiedler
    values produces an ordering where similar nodes are adjacent — exactly what
    we need. This is more principled than greedy chain-building.
    """
    print("\n[Method 7] Fiedler Spectral Ordering")
    t0 = time.time()

    # Build similarity from weight stats L2 distance (inverse, Gaussian kernel)
    # Gaussian kernel gives locality: nearby layers get high weight, far ones ≈ 0
    sigma = np.median(wf.D_l2_stats[wf.D_l2_stats > 0])
    S = np.exp(-wf.D_l2_stats ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(S, 0)  # no self-loops

    # Compute unnormalized graph Laplacian
    L = graph_laplacian(S, normed=False)

    # Fiedler vector = eigenvector for 2nd smallest eigenvalue
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    fiedler = eigenvectors[:, 1]  # 2nd column (eigenvalues are sorted ascending)

    # Sort shuffled indices by Fiedler values
    chain = list(np.argsort(fiedler))
    chain = orient_chain(chain, model)
    recovered = chain_to_permutation(chain, ground_truth_perm)

    elapsed = time.time() - t0
    metrics = evaluate(recovered, num_layers)
    print(f"  Kendall tau: {metrics['kendall_tau']:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
          f"Displacement: {metrics['displacement']:.2f}, Time: {elapsed:.1f}s")
    return {"method": "fiedler_spectral", "type": "math",
            "recovered": recovered, "time": elapsed, **metrics}


def method_causal_pairwise(model, num_layers, ground_truth_perm):
    """Recover order using pairwise causality tests with random activations.

    For each pair (i,j), pass random input through layer i then j, and compare
    the output norm to passing through j then i. In a trained transformer,
    layer k's output is better conditioned as input for layer k+1 than for an
    arbitrary other layer. The order that produces lower output norm / more
    stable activations is more likely the correct one.

    This gives a directed pairwise preference matrix, from which we extract
    a total ordering via win-counting (like a tournament).
    """
    print("\n[Method 8a] Causal Pairwise Test")
    t0 = time.time()

    torch.manual_seed(SEED)
    hidden_size = model.config.hidden_size
    fake_hidden = torch.randn(2, 32, hidden_size,
                              dtype=torch.bfloat16, device=DEVICE) * 0.1
    pos_ids = torch.arange(32, device=DEVICE).unsqueeze(0).expand(2, -1)
    position_embeddings = model.model.rotary_emb(fake_hidden, pos_ids)

    # For each pair, test both orderings
    wins = np.zeros((num_layers, num_layers))
    with torch.no_grad():
        # Pre-compute single-layer outputs for all layers
        single_outputs = {}
        for i in range(num_layers):
            layer = model.model.layers[i]
            out = layer(fake_hidden, position_ids=pos_ids,
                       position_embeddings=position_embeddings, use_cache=False)
            single_outputs[i] = out[0] if isinstance(out, tuple) else out

        for i in tqdm(range(num_layers), desc="  Causal pairs"):
            for j in range(i + 1, num_layers):
                # Order i->j: feed output of i into j
                out_ij = model.model.layers[j](
                    single_outputs[i], position_ids=pos_ids,
                    position_embeddings=position_embeddings, use_cache=False)
                h_ij = out_ij[0] if isinstance(out_ij, tuple) else out_ij

                # Order j->i: feed output of j into i
                out_ji = model.model.layers[i](
                    single_outputs[j], position_ids=pos_ids,
                    position_embeddings=position_embeddings, use_cache=False)
                h_ji = out_ji[0] if isinstance(out_ji, tuple) else out_ji

                # Lower output std = more stable = more natural order
                score_ij = h_ij.float().std().item()
                score_ji = h_ji.float().std().item()

                if score_ij <= score_ji:
                    wins[i, j] = 1  # i before j is more stable
                else:
                    wins[j, i] = 1

    total_wins = wins.sum(axis=1)
    order = list(np.argsort(-total_wins))
    chain = orient_chain(order, model)
    recovered = chain_to_permutation(chain, ground_truth_perm)

    elapsed = time.time() - t0
    metrics = evaluate(recovered, num_layers)
    print(f"  Kendall tau: {metrics['kendall_tau']:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
          f"Displacement: {metrics['displacement']:.2f}, Time: {elapsed:.1f}s")
    return {"method": "causal_pairwise", "type": "math",
            "recovered": recovered, "time": elapsed, **metrics}


def method_ensemble_rank(model, num_layers, ground_truth_perm, wf: WeightFeatures):
    """Ensemble: average rank position across multiple distance matrices.

    Each distance matrix produces a chain. For each shuffled layer, we average
    its position across all chains. The ensemble ranking is more robust than
    any single method because different metrics capture different aspects of
    the layer progression.
    """
    print("\n[Method 8] Ensemble Rank Averaging")
    t0 = time.time()

    distance_matrices = [wf.D_l2_stats, wf.D_l2_svd, wf.D_cosine_full, wf.D_l2_norm]
    all_chains = []

    for D in distance_matrices:
        chain = greedy_nearest_neighbor_chain(D)
        chain = two_opt_improve(D, chain)
        chain = orient_chain(chain, model)
        all_chains.append(chain)

    # For each shuffled index, compute its average position across chains
    avg_pos = np.zeros(num_layers)
    for chain in all_chains:
        for pos, idx in enumerate(chain):
            avg_pos[idx] += pos
    avg_pos /= len(all_chains)

    # Final ordering: sort shuffled indices by average position
    chain = list(np.argsort(avg_pos))
    recovered = chain_to_permutation(chain, ground_truth_perm)

    elapsed = time.time() - t0
    metrics = evaluate(recovered, num_layers)
    print(f"  Kendall tau: {metrics['kendall_tau']:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
          f"Displacement: {metrics['displacement']:.2f}, Time: {elapsed:.1f}s")
    return {"method": "ensemble_rank", "type": "math",
            "recovered": recovered, "time": elapsed, **metrics}


# ============================================================================
# Dataset-Based Methods
# ============================================================================

CALIB_DATA_DIR = Path(__file__).parent / "data"

def load_calibration_data():
    """Load pre-generated calibration data from disk.

    Run generate_calibration.py first to create the data.
    """
    calib_path = CALIB_DATA_DIR / "input_ids.pt"
    if not calib_path.exists():
        raise FileNotFoundError(
            f"Calibration data not found at {calib_path}. "
            "Run generate_calibration.py first:\n"
            "  poetry run python experiments/layer_shuffle_recovery/generate_calibration.py"
        )
    print(f"\nLoading calibration data from {CALIB_DATA_DIR}...")
    input_ids = torch.load(calib_path, weights_only=True).to(DEVICE)
    print(f"  Loaded: {input_ids.shape}")
    return input_ids


def _prepare_forward(model, input_ids):
    """Shared setup for manual layer-by-layer forward passes."""
    hidden = model.model.embed_tokens(input_ids)
    pos_ids = torch.arange(input_ids.shape[1], device=DEVICE).unsqueeze(0).expand(input_ids.shape[0], -1)
    position_embeddings = model.model.rotary_emb(hidden, pos_ids)
    return hidden, pos_ids, position_embeddings


def _run_single_layer(layer, hidden, pos_ids, position_embeddings):
    """Forward through a single decoder layer."""
    output = layer(hidden, position_ids=pos_ids,
                   position_embeddings=position_embeddings, use_cache=False)
    return output[0] if isinstance(output, tuple) else output


def _compute_lm_loss(model, hidden, input_ids):
    """Apply final norm + lm_head and compute next-token CE loss."""
    hidden = model.model.norm(hidden)
    logits = model.lm_head(hidden)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).item()


@torch.no_grad()
def compute_loss_with_order(model, input_ids, layer_order):
    """Full forward pass with layers executed in given order."""
    hidden, pos_ids, pos_emb = _prepare_forward(model, input_ids)
    for idx in layer_order:
        hidden = _run_single_layer(model.model.layers[idx], hidden, pos_ids, pos_emb)
    return _compute_lm_loss(model, hidden, input_ids)


@torch.no_grad()
def method_greedy_perplexity(model, num_layers, ground_truth_perm, input_ids):
    """Greedy layer-by-layer selection minimizing loss.

    Optimized: caches hidden state prefix so each candidate only runs 1 layer forward.
    Total: N + (N-1) + ... + 1 = N(N+1)/2 = 406 single-layer forwards for N=28.
    """
    print("\n[Method 9] Greedy Perplexity Ordering")
    t0 = time.time()

    sub_ids = input_ids[:16, :128]
    hidden_base, pos_ids, pos_emb = _prepare_forward(model, sub_ids)

    remaining = set(range(num_layers))
    order = []
    cached_hidden = hidden_base  # hidden state after all decided layers

    for pos in tqdm(range(num_layers), desc="  Greedy selection"):
        best_layer, best_loss = -1, float('inf')
        for candidate in remaining:
            # Only run 1 new layer on top of cached prefix
            h = _run_single_layer(model.model.layers[candidate], cached_hidden, pos_ids, pos_emb)
            loss = _compute_lm_loss(model, h, sub_ids)

            if loss < best_loss:
                best_loss = loss
                best_layer = candidate

        order.append(best_layer)
        remaining.remove(best_layer)
        # Update cache: run the chosen layer
        cached_hidden = _run_single_layer(model.model.layers[best_layer], cached_hidden, pos_ids, pos_emb)

        if pos % 7 == 0:
            print(f"    Position {pos}: selected shuffled layer {best_layer} (loss={best_loss:.4f})")

    recovered = chain_to_permutation(order, ground_truth_perm)
    elapsed = time.time() - t0
    metrics = evaluate(recovered, num_layers)
    print(f"  Kendall tau: {metrics['kendall_tau']:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
          f"Displacement: {metrics['displacement']:.2f}, Time: {elapsed:.1f}s")
    return {"method": "greedy_perplexity", "type": "dataset",
            "recovered": recovered, "time": elapsed, **metrics}


@torch.no_grad()
def method_pairwise_tournament(model, num_layers, ground_truth_perm, input_ids, seed_order):
    """Pairwise ordering: starting from a good seed, test each adjacent swap.

    Seeded with the best math method result so the model is nearly functional.
    Swapping two positions in a near-correct order gives a meaningful loss signal.
    Uses bubble-sort-style passes: repeatedly sweep and swap adjacent pairs if
    it reduces loss. Much more effective than random pair comparisons.
    """
    print("\n[Method 10] Pairwise Bubble Refinement")
    t0 = time.time()

    sub_ids = input_ids[:32, :196]
    order = list(seed_order)
    current_loss = compute_loss_with_order(model, sub_ids, order)
    print(f"  Seed loss: {current_loss:.4f}")

    # Bubble-sort passes: swap adjacent elements if it improves loss
    max_passes = 5
    total_swaps = 0
    for pass_num in range(max_passes):
        swaps = 0
        for i in range(num_layers - 1):
            trial = order[:]
            trial[i], trial[i + 1] = trial[i + 1], trial[i]
            trial_loss = compute_loss_with_order(model, sub_ids, trial)
            if trial_loss < current_loss - 1e-6:
                order = trial
                current_loss = trial_loss
                swaps += 1
        total_swaps += swaps
        print(f"    Pass {pass_num + 1}: {swaps} swaps, loss={current_loss:.4f}")
        if swaps == 0:
            break

    # Also try non-adjacent swaps for any remaining improvements
    improved = True
    while improved:
        improved = False
        for i in range(num_layers):
            for j in range(i + 2, num_layers):
                trial = order[:]
                trial[i], trial[j] = trial[j], trial[i]
                trial_loss = compute_loss_with_order(model, sub_ids, trial)
                if trial_loss < current_loss - 1e-6:
                    order = trial
                    current_loss = trial_loss
                    total_swaps += 1
                    improved = True

    recovered = chain_to_permutation(order, ground_truth_perm)
    elapsed = time.time() - t0
    metrics = evaluate(recovered, num_layers)
    print(f"  {total_swaps} total swaps, final loss: {current_loss:.4f}, Time: {elapsed:.1f}s")
    print(f"  Kendall tau: {metrics['kendall_tau']:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
          f"Displacement: {metrics['displacement']:.2f}")
    return {"method": "pairwise_bubble", "type": "dataset",
            "recovered": recovered, "time": elapsed, **metrics}


@torch.no_grad()
def method_simulated_annealing(model, num_layers, ground_truth_perm, input_ids, seed_order):
    """Simulated annealing seeded from best math result, refining with loss.

    Uses low temperature (seed is already close) and biases toward local swaps
    (adjacent/near swaps are 3x more likely than distant ones).
    """
    print(f"\n[Method 11] Simulated Annealing ({SA_ITERATIONS} iterations)")
    t0 = time.time()

    sub_ids = input_ids[:8, :128]

    current_order = list(seed_order)
    current_loss = compute_loss_with_order(model, sub_ids, current_order)
    best_order = current_order[:]
    best_loss = current_loss

    print(f"  Seed loss: {current_loss:.4f}")
    rng = np.random.RandomState(SEED)

    t_init = 0.3  # low temp: seed is already good, we're fine-tuning
    for it in tqdm(range(SA_ITERATIONS), desc="  SA iterations"):
        T = t_init * (1.0 - it / SA_ITERATIONS)

        # Bias toward local swaps: 75% adjacent/near, 25% distant
        i = rng.randint(0, num_layers)
        if rng.random() < 0.75:
            offset = rng.choice([-2, -1, 1, 2])
            j = (i + offset) % num_layers
        else:
            j = rng.randint(0, num_layers)
            while j == i:
                j = rng.randint(0, num_layers)

        new_order = current_order[:]
        new_order[i], new_order[j] = new_order[j], new_order[i]

        new_loss = compute_loss_with_order(model, sub_ids, new_order)
        delta = new_loss - current_loss

        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / max(T, 1e-10))):
            current_order = new_order
            current_loss = new_loss
            if current_loss < best_loss:
                best_loss = current_loss
                best_order = current_order[:]

        if it % 300 == 0:
            print(f"    Iter {it}: loss={current_loss:.4f}, best={best_loss:.4f}, T={T:.4f}")

    print(f"  Final best loss: {best_loss:.4f}")
    recovered = chain_to_permutation(best_order, ground_truth_perm)

    elapsed = time.time() - t0
    metrics = evaluate(recovered, num_layers)
    print(f"  Kendall tau: {metrics['kendall_tau']:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
          f"Displacement: {metrics['displacement']:.2f}, Time: {elapsed:.1f}s")
    return {"method": "simulated_annealing", "type": "dataset",
            "recovered": recovered, "time": elapsed, **metrics}


@torch.no_grad()
def method_remove_reinsert(model, num_layers, ground_truth_perm, input_ids, seed_order):
    """Refine seed ordering by removing each layer and reinserting at best position.

    Starting from the best math result, iterate: for each layer, remove it from
    the sequence, try all positions, reinsert at the one with lowest loss.
    Repeat passes until no improvement. This is a local search that can fix
    individual misplacements.
    """
    print("\n[Method 12] Remove-Reinsert Refinement")
    t0 = time.time()

    # Use more data for stronger signal on similar layers
    sub_ids = input_ids[:32, :196]
    order = list(seed_order)
    current_loss = compute_loss_with_order(model, sub_ids, order)
    print(f"  Seed loss: {current_loss:.4f}")

    max_passes = 3
    for pass_num in range(max_passes):
        moves = 0
        for idx in tqdm(range(num_layers), desc=f"  Pass {pass_num + 1}"):
            # Remove layer at current position
            layer_val = order[idx]
            remaining = order[:idx] + order[idx + 1:]

            # Try all insertion positions
            best_pos, best_loss = idx, current_loss
            for pos in range(num_layers):
                trial = remaining[:pos] + [layer_val] + remaining[pos:]
                loss = compute_loss_with_order(model, sub_ids, trial)
                if loss < best_loss - 1e-6:
                    best_loss = loss
                    best_pos = pos

            if best_pos != idx:
                order = remaining[:best_pos] + [layer_val] + remaining[best_pos:]
                current_loss = best_loss
                moves += 1

        print(f"    Pass {pass_num + 1}: {moves} moves, loss={current_loss:.4f}")
        if moves == 0:
            break

    recovered = chain_to_permutation(order, ground_truth_perm)
    elapsed = time.time() - t0
    metrics = evaluate(recovered, num_layers)
    print(f"  Final loss: {current_loss:.4f}, Time: {elapsed:.1f}s")
    print(f"  Kendall tau: {metrics['kendall_tau']:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
          f"Displacement: {metrics['displacement']:.2f}")
    return {"method": "remove_reinsert", "type": "dataset",
            "recovered": recovered, "time": elapsed, **metrics}


# ============================================================================
# Visualization
# ============================================================================

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _color_bar(value, max_val=1.0, width=20):
    """Render a colored progress bar."""
    filled = int(value / max_val * width)
    filled = max(0, min(width, filled))
    if value >= 0.8:
        color = GREEN
    elif value >= 0.4:
        color = YELLOW
    else:
        color = RED
    return f"{color}{'█' * filled}{DIM}{'░' * (width - filled)}{RESET}"


def _render_layer_row(recovered, num_layers):
    """Render a row showing each layer position: green if correct, red if wrong."""
    ideal = list(range(num_layers))
    cells = []
    for i in range(num_layers):
        if recovered[i] == ideal[i]:
            cells.append(f"{GREEN}{recovered[i]:>2}{RESET}")
        else:
            cells.append(f"{RED}{recovered[i]:>2}{RESET}")
    return " ".join(cells)


def print_results(results, num_layers, ground_truth_perm):
    """Print a rich terminal visualization of all method results."""
    ideal = list(range(num_layers))
    sorted_results = sorted(results, key=lambda x: -x['kendall_tau'])

    width = 90

    print()
    print(f"{BOLD}{'═' * width}{RESET}")
    print(f"{BOLD}  LAYER SHUFFLE RECOVERY RESULTS{RESET}")
    print(f"{BOLD}{'═' * width}{RESET}")

    # Ground truth reference row
    print(f"\n  {DIM}Ground truth (ideal):{RESET}")
    print(f"  {DIM}{'  '.join(f'{i:>2}' for i in ideal)}{RESET}")
    print()

    for r in sorted_results:
        recovered = r['recovered']
        tau = r['kendall_tau']
        acc = r['accuracy']
        disp = r['displacement']
        correct = sum(1 for i in range(num_layers) if recovered[i] == i)

        # Method header
        type_badge = f"{CYAN}[math]{RESET}" if r['type'] == 'math' else f"{YELLOW}[data]{RESET}"
        print(f"  {'─' * width}")
        print(f"  {BOLD}{r['method']}{RESET}  {type_badge}  {DIM}{r['time']:.1f}s{RESET}")
        print()

        # Layer positions row
        print(f"  {_render_layer_row(recovered, num_layers)}")
        print()

        # Metrics with bars
        print(f"    Kendall τ   {_color_bar(max(0, tau))}  {tau:+.4f}")
        print(f"    Accuracy    {_color_bar(acc)}  {correct}/{num_layers} correct")
        print(f"    Displacement{_color_bar(1 - disp / num_layers)}  {disp:.2f} avg positions off")
        print()

    print(f"  {'═' * width}")

    # Summary table
    print(f"\n  {BOLD}{'Method':<30} {'Type':<6} {'τ':>7} {'Acc':>7} {'Disp':>6} {'Time':>7}{RESET}")
    print(f"  {'─' * 67}")
    for r in sorted_results:
        tau = r['kendall_tau']
        if tau >= 0.8:
            tc = GREEN
        elif tau >= 0.4:
            tc = YELLOW
        else:
            tc = RED
        print(f"  {r['method']:<30} {r['type']:<6} {tc}{tau:>+.4f}{RESET} "
              f"{r['accuracy']:>6.1%} {r['displacement']:>5.2f} {r['time']:>6.1f}s")
    print(f"  {'─' * 67}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model, num_layers = load_model()
    ground_truth_perm = shuffle_layers(model, num_layers)

    # Precompute shared weight features
    wf = WeightFeatures(model, num_layers)

    # ---- Math-only methods ----
    results = []
    results.append(method_weight_stats_continuity(model, num_layers, ground_truth_perm, wf))
    results.append(method_svd_spectrum(model, num_layers, ground_truth_perm, wf))
    results.append(method_layernorm_progression(model, num_layers, ground_truth_perm, wf))
    results.append(method_tsp_full_weights(model, num_layers, ground_truth_perm, wf))
    results.append(method_greedy_cosine(model, num_layers, ground_truth_perm, wf))
    results.append(method_activation_flow(model, num_layers, ground_truth_perm))
    results.append(method_fiedler_spectral(model, num_layers, ground_truth_perm, wf))
    results.append(method_causal_pairwise(model, num_layers, ground_truth_perm))
    results.append(method_ensemble_rank(model, num_layers, ground_truth_perm, wf))

    # ---- Dataset-based methods ----
    input_ids = load_calibration_data()

    # Find best math ordering to seed dataset methods
    best_math = max(results, key=lambda r: r['kendall_tau'])
    best_math_chain = best_math['recovered']
    # Convert recovered (original layer ids) back to shuffled-space order
    inv_perm = {int(ground_truth_perm[s]): s for s in range(num_layers)}
    best_math_order = [inv_perm[orig] for orig in best_math_chain]
    print(f"\nSeeding dataset methods with best math result: {best_math['method']} (τ={best_math['kendall_tau']:.4f})")

    results.append(method_greedy_perplexity(model, num_layers, ground_truth_perm, input_ids))
    results.append(method_pairwise_tournament(model, num_layers, ground_truth_perm, input_ids, best_math_order))
    results.append(method_simulated_annealing(model, num_layers, ground_truth_perm, input_ids, best_math_order))
    results.append(method_remove_reinsert(model, num_layers, ground_truth_perm, input_ids, best_math_order))

    # ---- Pretty visualization ----
    print_results(results, num_layers, ground_truth_perm)

    total_time = sum(r['time'] for r in results)
    print(f"\nTotal time (methods only): {total_time:.1f}s")

    output = {
        "model": MODEL_NAME,
        "seed": SEED,
        "num_layers": num_layers,
        "ground_truth_perm": ground_truth_perm.tolist(),
        "results": results,
    }
    out_path = RESULTS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")
