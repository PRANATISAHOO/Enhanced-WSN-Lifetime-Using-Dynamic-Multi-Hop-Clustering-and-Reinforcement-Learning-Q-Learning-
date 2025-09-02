

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

# Load dataset
dataset = pd.read_csv("wsn_real_data.csv")
nodes = dataset[['X', 'Y']].values
energy = dataset['Energy'].values.copy()

num_nodes = len(nodes)
base_station = np.array([500, 1050])

# Parameters
packet_size = 4000
E_elec = 50e-9
E_fs = 10e-12
E_mp = 0.0013e-12
d0 = 87
p = 0.05
rounds = 50

# Q-learning Parameters
alpha, beta, gamma_w = 0.6, 0.3, 0.1
learning_rate = 0.1
discount = 0.9
Q_table = {}

# Metrics
pdr_list, throughput_list = [], []
ch_lifetime = np.zeros(num_nodes)
total_energy_list = []
alive_nodes_list = []

# Functions
def energy_cost(d):
    return packet_size * (E_elec + (E_fs * d**2 if d < d0 else E_mp * d**4))

def get_density(radius=100):
    density = np.zeros(num_nodes)
    for i in range(num_nodes):
        dist = np.linalg.norm(nodes - nodes[i], axis=1)
        density[i] = np.sum(dist < radius)
    return density

def select_cluster_heads(alive_nodes):
    density = get_density()
    dist_to_bs = np.linalg.norm(nodes[alive_nodes] - base_station, axis=1)
    score = alpha * energy[alive_nodes] + beta * density[alive_nodes] - gamma_w * dist_to_bs
    num_ch = max(1, int(p * len(alive_nodes)))
    chs = alive_nodes[np.argsort(score)[-num_ch:]]
    for ch in chs: ch_lifetime[ch] += 1
    return chs

def choose_next_hop(ch, chs):
    if ch not in Q_table:
        Q_table[ch] = {nh: 0 for nh in chs if nh != ch}
        Q_table[ch]['BS'] = 0
    return max(Q_table[ch], key=Q_table[ch].get)

def update_q(ch, next_hop, reward):
    max_q_next = max(Q_table[next_hop].values()) if next_hop in Q_table else 0
    Q_table[ch][next_hop] += learning_rate * (reward + discount * max_q_next - Q_table[ch][next_hop])

# Simulation
plt.ion()
for r in range(rounds):
    plt.clf()
    alive = np.where(energy > 0)[0]
    if len(alive) == 0: break

    chs = select_cluster_heads(alive)
    kdtree = KDTree(nodes[chs])
    ch_assignments = kdtree.query(nodes[alive], return_distance=False).flatten()

    delivered_packets = 0
    bits_transmitted = 0

    for i, node in enumerate(alive):
        ch = chs[ch_assignments[i]]
        if node != ch:
            d = np.linalg.norm(nodes[node] - nodes[ch])
            energy[node] = max(0, energy[node] - energy_cost(d))
            bits_transmitted += packet_size
            delivered_packets += 1
            plt.plot([nodes[node][0], nodes[ch][0]], [nodes[node][1], nodes[ch][1]],
                     color='skyblue', linewidth=0.4, alpha=0.5)

    for ch in chs:
        next_hop = choose_next_hop(ch, chs)
        if next_hop == 'BS':
            d = np.linalg.norm(nodes[ch] - base_station)
            energy[ch] = max(0, energy[ch] - energy_cost(d))
            reward = -energy_cost(d)
            plt.plot([nodes[ch][0], base_station[0]], [nodes[ch][1], base_station[1]],
                     color='limegreen', linestyle='--', linewidth=1.2, alpha=0.8)
        else:
            d = np.linalg.norm(nodes[ch] - nodes[next_hop])
            energy[ch] = max(0, energy[ch] - energy_cost(d))
            reward = -energy_cost(d)
            plt.plot([nodes[ch][0], nodes[next_hop][0]], [nodes[ch][1], nodes[next_hop][1]],
                     color='mediumorchid', linestyle='--', linewidth=1, alpha=0.7)
        update_q(ch, next_hop, reward)

    alive_nodes = np.where(energy > 0)[0]
    plt.scatter(nodes[alive_nodes, 0], nodes[alive_nodes, 1], s=15, c='dodgerblue', label='Alive Nodes', alpha=0.6)
    plt.scatter(nodes[chs, 0], nodes[chs, 1], s=70, c='crimson', label='Cluster Heads', edgecolors='black')
    plt.scatter(base_station[0], base_station[1], s=120, c='gold', marker='*', label='Base Station', edgecolors='black')

    for ch in chs:
        plt.plot([nodes[ch][0], base_station[0]], [nodes[ch][1], base_station[1]],
                 linestyle=':', linewidth=0.5, color='gray', alpha=0.4)

    plt.text(10, 1070, f"Round {r+1}", fontsize=10, bbox=dict(facecolor='white', alpha=0.6))

    plt.title("WSN Round Visualization", fontsize=14)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.pause(0.4)

    # Metrics update
    pdr_list.append(delivered_packets / len(alive))
    throughput_list.append(bits_transmitted)
    total_energy_list.append(np.sum(energy))
    alive_nodes_list.append(len(alive_nodes))

plt.ioff()
plt.show()

# Final Graphs
plt.figure(figsize=(8, 6))
plt.plot(pdr_list, label="PDR")
plt.title("Packet Delivery Ratio")
plt.xlabel("Round")
plt.ylabel("PDR")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(throughput_list, label="Throughput", color='orange')
plt.title("Throughput per Round")
plt.xlabel("Round")
plt.ylabel("Throughput (bits)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(range(num_nodes), ch_lifetime, color='purple')
plt.title("CH Lifetime (Times Selected)")
plt.xlabel("Node ID")
plt.ylabel("CH Count")
plt.grid(True)
plt.tight_layout()
plt.show()

heatmap_size = int(np.ceil(np.sqrt(len(energy))))
padded_energy = np.zeros((heatmap_size**2,))
padded_energy[:len(energy)] = energy
energy_matrix = padded_energy.reshape((heatmap_size, heatmap_size))

plt.figure(figsize=(8, 6))
plt.imshow(energy_matrix, cmap='hot', interpolation='nearest')
plt.colorbar(label='Residual Energy')
plt.title("Final Residual Energy Heatmap")
plt.axis('off')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(total_energy_list, label="Total Energy", color='red')
plt.title("Total Network Energy Over Time")
plt.xlabel("Round")
plt.ylabel("Energy (J)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(alive_nodes_list, label="Alive Nodes", color='blue')
plt.title("Alive Nodes Over Rounds")
plt.xlabel("Round")
plt.ylabel("Number of Alive Nodes")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(nodes[:, 0], nodes[:, 1], c=energy, cmap='hot', s=10)
plt.colorbar(label='Residual Energy (Joules)')
plt.title("Improved Final Residual Energy Heatmap (Scatter Style)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.tight_layout()
plt.show()

single_hop_energy = [total_energy_list[0] * (1 - 0.01 * i) for i in range(rounds)]

plt.figure(figsize=(10, 6))
plt.plot(total_energy_list, label="DC-MHRM + Q-Learning (Multi-Hop)", linewidth=2)
plt.plot(single_hop_energy, label="Baseline Single-Hop (Placeholder)", linestyle='--', linewidth=2)
plt.title("Total Network Energy: Multi-Hop vs Single-Hop")
plt.xlabel("Round")
plt.ylabel("Total Residual Energy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
import csv

# 1. PDR (Packet Delivery Ratio)
pd.DataFrame({
    "Round": list(range(1, len(pdr_list) + 1)),
    "PDR": pdr_list
}).to_csv("output_PDR.csv", index=False)

# 2. Throughput
pd.DataFrame({
    "Round": list(range(1, len(throughput_list) + 1)),
    "Throughput (bits)": throughput_list
}).to_csv("output_Throughput.csv", index=False)

# 3. CH Lifetime
pd.DataFrame({
    "Node ID": list(range(num_nodes)),
    "CH Count": ch_lifetime
}).to_csv("output_CH_Lifetime.csv", index=False)

# 4. Final Residual Energy Heatmap (Matrix Style)
heatmap_size = int(np.ceil(np.sqrt(len(energy))))
padded_energy = np.zeros((heatmap_size**2,))
padded_energy[:len(energy)] = energy
energy_matrix = padded_energy.reshape((heatmap_size, heatmap_size))
pd.DataFrame(energy_matrix).to_csv("output_Energy_Heatmap_Matrix.csv", index=False, header=False)

# 5. Energy Decay Over Time
pd.DataFrame({
    "Round": list(range(1, len(total_energy_list) + 1)),
    "Total Energy (J)": total_energy_list
}).to_csv("output_Total_Energy.csv", index=False)

# 6. Alive Nodes Over Time
pd.DataFrame({
    "Round": list(range(1, len(alive_nodes_list) + 1)),
    "Alive Nodes": alive_nodes_list
}).to_csv("output_Alive_Nodes.csv", index=False)

# 7. Improved Final Energy Heatmap (Scatter Style)
pd.DataFrame({
    "X": nodes[:, 0],
    "Y": nodes[:, 1],
    "Residual Energy (J)": energy
}).to_csv("output_Energy_Heatmap_Scatter.csv", index=False)

# 8. Energy Comparison (Multi-hop vs Single-hop)
pd.DataFrame({
    "Round": list(range(1, rounds + 1)),
    "Multi-Hop (DC-MHRM + Q-Learning)": total_energy_list,
    "Single-Hop (Baseline Placeholder)": single_hop_energy
}).to_csv("output_Energy_Comparison.csv", index=False)