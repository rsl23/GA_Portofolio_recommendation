"""Genetic operators: selection, crossover and mutation."""

from __future__ import annotations

import numpy as np


def _rng(rng):
    return rng if rng is not None else np.random.default_rng()


def tournament_select(population, k: int = 5, rng=None):
    """Pick ``k`` random individuals and return the one with the best fitness."""
    rng = _rng(rng)
    picks = rng.integers(0, len(population), size=k)
    best = None
    for p in picks:
        c = population[int(p)]
        if c.fitness is not None and (best is None or c.fitness > best.fitness):
            best = c
    return best


def two_point_crossover(parent_a, parent_b, rng=None):
    """Swap the gene segment between two random cut points; return two children."""
    rng = _rng(rng)
    n = len(parent_a.genes)
    if n >= 2:
        i, j = sorted(rng.integers(0, n, size=2).tolist())
        if i == j:
            j = min(n - 1, i + 1)
    else:
        i = j = 0

    ga = list(parent_a.genes)
    gb = list(parent_b.genes)
    ga[i:j], gb[i:j] = gb[i:j], ga[i:j]

    child_a = parent_a.clone()
    child_b = parent_b.clone()
    child_a.genes = ga
    child_b.genes = gb
    child_a.fitness = None
    child_b.fitness = None
    return child_a, child_b

# Ini untuk crossover yang menggunakan saham" yg aktif saja letak titik potongnya
# def two_point_crossover(parent_a, parent_b, rng=None):
#     """
#     Smart Two-Point Crossover: 
#     Menjamin segmen yang ditukar mengandung gen (saham) yang aktif.
#     """
#     rng = _rng(rng)
#     n = len(parent_a.genes)
    
#     # 1. CARI "HOT ZONE" (Lokasi saham yang diisi uang oleh Parent A atau B)
#     active_indices = [
#         idx for idx in range(n) 
#         if parent_a.genes[idx] > 1e-6 or parent_b.genes[idx] > 1e-6
#     ]
    
#     # 2. PENENTUAN TITIK POTONG CERDAS
#     if len(active_indices) >= 2:
#         # Pilih 2 lokasi aktif secara acak sebagai titik potong
#         pt1 = int(rng.choice(active_indices))
#         pt2 = int(rng.choice(active_indices))
        
#         i, j = sorted([pt1, pt2])
#         # Lebarkan titik j sedikit agar saham di titik j ikut tertukar
#         j = min(n, j + 1) 
#     else:
#         # Fallback (Jaga-jaga) menggunakan cara acak normal
#         i, j = sorted(rng.integers(0, n, size=2).tolist())
#         if i == j:
#             j = min(n - 1, i + 1)

#     # 3. PROSES KAWIN SILANG (Murni Two-Point Crossover)
#     ga = list(parent_a.genes)
#     gb = list(parent_b.genes)
#     ga[i:j], gb[i:j] = gb[i:j], ga[i:j]

#     # 4. PEMBENTUKAN ANAK KROMOSOM
#     child_a = parent_a.clone()
#     child_b = parent_b.clone()
#     child_a.genes = ga
#     child_b.genes = gb
#     child_a.fitness = None
#     child_b.fitness = None
    
#     return child_a, child_b


def adaptive_hybrid_mutation(chromosome, generation: int, max_generation: int, rng=None):
    """Mutation rate decays ~15% -> ~1%; each mutation is either a small creep
    or a random reset (which may drop a stock - protected at the minimum)."""
    rng = _rng(rng)
    cfg = chromosome.config
    if max_generation <= 1:
        rate = cfg.mutation_rate_end
    else:
        rate = cfg.mutation_rate_start - (
            cfg.mutation_rate_start - cfg.mutation_rate_end
        ) * (generation / (max_generation - 1))
    rate = float(np.clip(rate, cfg.mutation_rate_end, cfg.mutation_rate_start))

    genes = list(chromosome.genes)
    n = len(genes)
    for i in range(n):
        if rng.random() > rate:
            continue
        if rng.random() < cfg.creep_prob:
            # fine-grained rebalancing
            genes[i] = float(np.clip(genes[i] + rng.normal(0.0, 0.05), 0.0, 1.0))
        else:
            active = sum(1 for g in genes if g > 1e-6)
            if genes[i] > 1e-6 and active > cfg.min_stocks:
                genes[i] = 0.0          # random reset: eliminate this stock
            else:
                genes[i] = float(rng.uniform(0.05, 1.0))  # reset to a fresh weight

    chromosome.genes = genes
    chromosome.normalize()
    chromosome.fitness = None
