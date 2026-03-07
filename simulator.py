"""
Adaptive-network SIR epidemic simulator.
Implements the exact model specification from project.md.
"""

import numpy as np


def simulate(beta, gamma, rho, N=200, p_edge=0.05, n_infected0=5, T=200, rng=None):
    """Run one replicate of the adaptive-network SIR model.

    Returns:
        infected_fraction: array of shape (T+1,) with fraction infected at each step
        rewire_counts: array of shape (T+1,) with rewiring events per step (0 at t=0)
        degree_histogram: array of shape (31,) with node counts per degree at final time
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- Initial network: Erdos-Renyi G(N, p) ---
    # Use adjacency sets for efficient neighbor lookups
    neighbors = [set() for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < p_edge:
                neighbors[i].add(j)
                neighbors[j].add(i)

    # --- Initial health state ---
    # 0=S, 1=I, 2=R
    state = np.zeros(N, dtype=np.int8)
    initial_infected = rng.choice(N, size=n_infected0, replace=False)
    state[initial_infected] = 1

    infected_fraction = np.zeros(T + 1)
    rewire_counts = np.zeros(T + 1, dtype=np.int64)
    infected_fraction[0] = np.sum(state == 1) / N

    for t in range(1, T + 1):
        # --- Step 1: Infection (synchronous) ---
        new_infections = set()
        infected_nodes = np.where(state == 1)[0]

        for i in infected_nodes:
            for j in neighbors[i]:
                if state[j] == 0:  # susceptible
                    if rng.random() < beta:
                        new_infections.add(j)

        # Apply infections
        for j in new_infections:
            state[j] = 1

        # --- Step 2: Recovery ---
        # Recompute infected set (includes newly infected)
        infected_nodes = np.where(state == 1)[0]
        for i in infected_nodes:
            if rng.random() < gamma:
                state[i] = 2

        # --- Step 3: Network rewiring ---
        rewire_count = 0
        # Collect all S-I edges (using state after infection and recovery)
        si_edges = []
        for i in range(N):
            if state[i] == 0:  # susceptible
                for j in neighbors[i]:
                    if state[j] == 1:  # infected neighbor
                        si_edges.append((i, j))

        # Process rewiring
        for s_node, i_node in si_edges:
            if rng.random() < rho:
                # Verify edge still exists (may have been removed by earlier rewiring)
                if i_node not in neighbors[s_node]:
                    continue
                # Remove edge
                neighbors[s_node].discard(i_node)
                neighbors[i_node].discard(s_node)
                # Find valid new partners: not self, not already connected
                candidates = []
                for k in range(N):
                    if k != s_node and k not in neighbors[s_node]:
                        candidates.append(k)
                if candidates:
                    new_partner = rng.choice(candidates)
                    neighbors[s_node].add(new_partner)
                    neighbors[new_partner].add(s_node)
                    rewire_count += 1

        infected_fraction[t] = np.sum(state == 1) / N
        rewire_counts[t] = rewire_count

    # --- Final degree histogram ---
    degree_histogram = np.zeros(31, dtype=np.int64)
    for i in range(N):
        deg = min(len(neighbors[i]), 30)
        degree_histogram[deg] += 1

    return infected_fraction, rewire_counts, degree_histogram
