import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from cmath import sqrt, pi
from collections import Counter
from operator import add
from functools import reduce, partial
from p_tqdm import p_map
from scipy.optimize import fsolve
from multiprocessing import set_start_method

set_start_method("spawn")


def gen_random_graph_with_degree_dist(N, p):
    G = nx.erdos_renyi_graph(N, p)
    degree_seq = sorted([d for n, d in G.degree()], reverse=True)
    degree_count = Counter(degree_seq)
    return G, degree_count


def get_connected_components_dist(N, p):
    G, _ = gen_random_graph_with_degree_dist(N, p)
    return Counter([len(cc) for cc in sorted(nx.connected_components(G), key=len, reverse=True)])


def get_connected_components_dist_with_aggregation(N, k, nsamples, parallel=True):
    p = k * 1. / (N - 1)
    if parallel:
        cumulative_dist = reduce(add, p_map(get_connected_components_dist, [N] * nsamples, [p] * nsamples))
    else:
        cumulative_dist = reduce(add, [get_connected_components_dist(N, p) for _ in range(nsamples)])
    cc, count = zip(*sorted(cumulative_dist.items()))
    count = [float(x) / nsamples for x in count]
    return cc, count


def get_largest_connected_component_size_with_aggregation(N, k, nsamples):
    cc, count = get_connected_components_dist_with_aggregation(N, k, nsamples, parallel=False)
    return cc[-1]


def review_connected_components_size_dist(N, k, nsamples, ax):
    cc, count = get_connected_components_dist_with_aggregation(N, k, nsamples)
    ax.loglog(cc, count, label=f"{N=}; {nsamples=}")


def review_connected_components_theoretical_size(N, ax):
    ccs = list(range(1, N))
    sizes = [1 / (sqrt(2 * pi) * m ** (2.5)) for m in ccs]
    ax.loglog(ccs, sizes, label=f"theoretical;")


def review_connected_components_size_dist_complete(k):
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(1, 1, 1)
    review_connected_components_size_dist(101, k, 10000, ax)
    review_connected_components_size_dist(1001, k, 1000, ax)
    review_connected_components_size_dist(10001, k, 100, ax)
    review_connected_components_theoretical_size(10001, ax)

    ax.set_title("Connected components size distribution")
    ax.set_xlabel("log[connected_component]")
    ax.set_ylabel("log[size]")
    ax.legend()
    # plt.show()
    fig.savefig(f"images/2_{k=}.jpg")


def review_larges_connected_component_size(N, nsamples, ax):
    ks = np.arange(0., 2.01, 0.01)
    lcc_sizes = p_map(get_largest_connected_component_size_with_aggregation, [N] * len(ks), ks, [nsamples] * len(ks))
    lcc_sizes = [s * 1. / N for s in lcc_sizes]
    ax.plot(ks, lcc_sizes, label=f"{N=}")


def theoretical_largest_component_part(g, k):
    return 1 - g - np.exp(-g * k)


def review_theoretical_largest_component_part(ax):
    ks = np.arange(0., 2.01, 0.01)
    fs = [partial(theoretical_largest_component_part, k=k) for k in ks]
    gs = p_map(partial(fsolve, x0=0.5), fs)
    gs = [g[0] for g in gs]
    ax.plot(ks, gs, label="theoretical")


def review_larges_connected_component_size_complete():
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(1, 1, 1)
    review_larges_connected_component_size(101, 5000, ax)
    review_larges_connected_component_size(301, 500, ax)
    review_larges_connected_component_size(1001, 100, ax)
    review_theoretical_largest_component_part(ax)

    ax.set_title("Largest connected components part of graph distribution")
    ax.set_xlabel("log[k]")
    ax.set_ylabel("log[size / N]")
    ax.legend()
    fig.savefig(f"images/3.jpg")


def task_2():
    for k in [0.5, 0.9, 0.98, 1.0, 1.02, 1.1, 2.0]:
        review_connected_components_size_dist_complete(k)


flat_map = lambda f, xs: (y for ys in xs for y in f(ys))


def remove_all_cycles(G):
    c = nx.cycle_basis(G)
    c_nodes = list(set(flat_map(lambda _: _, c)))
    G.remove_nodes_from(c_nodes)

def get_part_of_graph_without_cycles(G, N):
    """Graph will mutate after"""
    remove_all_cycles(G)
    return len(G.nodes) / N

def get_average_part_of_graph_without_cycles(N, k, nsamples):
    p = k * 1. / (N - 1)
    graphs = [nx.erdos_renyi_graph(N, p) for _ in range(nsamples)]
    average_part = np.mean([get_part_of_graph_without_cycles(g, N) for g in graphs])
    return average_part

def review_average_part_of_graph_without_cycles(N, nsamples, ax):
    ks = np.arange(0., 2.01, 0.01)
    parts = p_map(get_average_part_of_graph_without_cycles, [N] * len(ks), ks, [nsamples] * len(ks))
    ax.plot(ks, parts, label=f"{N=}")

def review_average_part_of_graph_without_cycles_complete():
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(1, 1, 1)
    review_average_part_of_graph_without_cycles(101, 1000, ax)
    review_average_part_of_graph_without_cycles(301, 250, ax)
    review_average_part_of_graph_without_cycles(1001, 50, ax)

    ax.set_title("Average part of graph without cycles distribution")
    ax.set_xlabel("k")
    ax.set_ylabel("size / N")
    ax.legend()
    fig.savefig(f"images/4.jpg")



if __name__ == '__main__':
    review_average_part_of_graph_without_cycles_complete()
