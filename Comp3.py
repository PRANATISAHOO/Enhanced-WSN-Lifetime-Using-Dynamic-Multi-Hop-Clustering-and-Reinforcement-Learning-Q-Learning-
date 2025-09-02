import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN

# Load dataset
data = pd.read_csv("wsn_real_data.csv")
nodes = data[['X', 'Y']].values
energy_init = data['Energy'].values.copy()

# Parameters
num_nodes = nodes.shape[0]
base_station = np.array([500, 1050])
rounds = 50
packet_size = 4000
E_elec = 50e-9
E_fs = 10e-12
E_mp = 0.0013e-12
d0 = 87

# Helper energy function
def energy_cost(d):
    if d < d0:
        return packet_size * (E_elec + E_fs * d ** 2)
    else:
        return packet_size * (E_elec + E_mp * d ** 4)

# Initialize metric storage
results = {proto: {'alive': [], 'energy': [], 'ch_count': [], 'latency': [], 'energy_std': [], 'throughput': [], 'pdr': [], 'ch_load': []} for proto in [
    'LEACH', 'DBSCAN-LEACH', 'PEGASIS', 'SEP', 'TEEN', 'HEED', 'MSOFSO', 'GWW', 'Proposed']}

# Protocol simulation templates (simplified)
def simulate_protocol(name):
    np.random.seed(0)
    energy = energy_init.copy()
    alive = np.ones(num_nodes, dtype=bool)
    for r in range(rounds):
        alive_nodes = np.where(energy > 0)[0]
        if len(alive_nodes) == 0:
            break

        # ----- CH Selection -----
        if name == 'LEACH':
            p = 0.05
            chs = np.random.choice(alive_nodes, int(p * len(alive_nodes)), replace=False)

        elif name == 'DBSCAN-LEACH':
            db = DBSCAN(eps=100, min_samples=5).fit(nodes[alive_nodes])
            chs = []
            for lbl in set(db.labels_):
                if lbl != -1:
                    cluster = alive_nodes[db.labels_ == lbl]
                    centroid = nodes[cluster].mean(axis=0)
                    ch_idx = cluster[np.argmin(np.linalg.norm(nodes[cluster] - centroid, axis=1))]
                    chs.append(ch_idx)
            chs = np.array(chs)

        elif name == 'PEGASIS':
            chs = [alive_nodes[0]]

        elif name == 'SEP': 
            prob = energy[alive_nodes] / energy[alive_nodes].max()  # proBABILITY
            chs = alive_nodes[np.random.rand(len(alive_nodes)) < prob * 0.05]  #CH

        elif name == 'TEEN': #threshold sensitive energy efficient network protocol
            chs = alive_nodes[np.random.rand(len(alive_nodes)) < 0.04]

        elif name == 'HEED': #hybrid energy efficient destributed clustering
            weights = energy[alive_nodes] / np.linalg.norm(nodes[alive_nodes] - base_station, axis=1)
            chs = alive_nodes[np.argsort(weights)[-int(0.05 * len(alive_nodes)):]]

        elif name == 'MSOFSO': #modified self organized fuzzy snake optimizer
            chs = alive_nodes[np.random.choice(len(alive_nodes), int(0.06 * len(alive_nodes)), replace=False)]

        elif name == 'GWW':  #gray wolf and whale optimization hybrid
            dist_bs = np.linalg.norm(nodes[alive_nodes] - base_station, axis=1)
            chs = alive_nodes[np.argsort(dist_bs)[:int(0.05 * len(alive_nodes))]]

        elif name == 'Proposed':
            dist_bs = np.linalg.norm(nodes[alive_nodes] - base_station, axis=1)
            scores = 0.6 * (energy[alive_nodes] / energy[alive_nodes].max()) - \
                     0.3 * (dist_bs / dist_bs.max()) + \
                     0.1 * np.random.rand(len(alive_nodes))
            chs = alive_nodes[np.argsort(scores)[-int(0.05 * len(alive_nodes)):]]

        else:
            chs = []

        total_energy_used = 0
        latencies = []
        packets_sent = 0
        ch_loads = {ch: 0 for ch in chs}

        for i in alive_nodes:
            if i in chs:
                d = np.linalg.norm(nodes[i] - base_station)
            else:
                if len(chs) > 0:
                    dists = np.linalg.norm(nodes[i] - nodes[chs], axis=1)
                    d = np.min(dists)
                    closest_ch = chs[np.argmin(dists)]
                    ch_loads[closest_ch] += 1
                else:
                    d = 0

            e = energy_cost(d)
            energy[i] -= e
            total_energy_used += e
            if d > 0:
                latencies.append(d / 2e8)
                packets_sent += 1
            if energy[i] <= 0:
                alive[i] = False

        results[name]['alive'].append(np.sum(alive))
        results[name]['energy'].append(np.sum(energy[energy > 0]))
        results[name]['ch_count'].append(len(chs))
        results[name]['latency'].append(np.mean(latencies) if latencies else 0)
        results[name]['energy_std'].append(np.std(energy[alive]))
        results[name]['throughput'].append(packets_sent)
        results[name]['pdr'].append(packets_sent / len(alive_nodes) if len(alive_nodes) > 0 else 0)
        results[name]['ch_load'].append(np.mean(list(ch_loads.values())) if ch_loads else 0)

# Simulate all protocols
for proto in results:
    simulate_protocol(proto)

# Plotting function
def plot_metric(metric_key, title, ylabel):
    plt.figure(figsize=(12, 6))
    for proto in results:
        plt.plot(results[proto][metric_key], label=proto)
    plt.title(title)
    plt.xlabel("Rounds")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Generate plots
plot_metric('alive', "Alive Nodes over Rounds", "Alive Nodes")
plot_metric('energy', "Total Energy over Rounds", "Remaining Energy (J)")
plot_metric('ch_count', "Cluster Head Count over Rounds", "CH Count")
plot_metric('latency', "Average Latency over Rounds", "Latency (s)")
plot_metric('energy_std', "Energy Standard Deviation over Rounds", "Std Dev of Energy")
plot_metric('throughput', "Throughput over Rounds", "Packets Delivered")
plot_metric('pdr', "Packet Delivery Ratio over Rounds", "PDR")
plot_metric('ch_load', "Cluster Head Load over Rounds", "Avg CH Load")

# Export to CSV
final_metrics = {}
for proto in results:
    final_metrics[proto] = {
        'Avg. Packet Delivery Ratio (%)': 100 * np.mean(results[proto]['pdr']),
        'Avg. Throughput (kbps)': np.mean(results[proto]['throughput']) * packet_size * 8 / 1e3,
        'First Node Dead (FND) Round': next((i for i, v in enumerate(results[proto]['alive']) if v < num_nodes), 0),
        'Last Node Dead (LND) Round': next((i for i, v in enumerate(results[proto]['alive'][::-1]) if v > 0), 0),
        'Total Energy Consumption (J)': energy_init.sum() - results[proto]['energy'][-1],
        'Residual Energy (J)': results[proto]['energy'][-1],
        'Energy Standard Deviation': results[proto]['energy_std'][-1],
        'Network Lifetime (Rounds)': len(results[proto]['alive'])
    }

metrics_df = pd.DataFrame(final_metrics).T
metrics_df.to_csv("protocol_comparison_metrics.csv")

# Note: Qualitative Ratings (like CH Accuracy, Routing Efficiency, etc.) should ideally be explained
# in an appendix or methodology section of your paper, referencing the logic used in code.
