import networkx as nx
import matplotlib.pyplot as plt
import multiprocessing
import time


# ---------------------------------------------------------
# GLOBAL VARIABLE USED BY PARALLEL WORKERS
# ---------------------------------------------------------

graph = None


# ---------------------------------------------------------
# CREATE NETWORK
# ---------------------------------------------------------

def create_network(number_of_nodes, connection_probability=0.05):
    """
    Creates a random computer network.

    number_of_nodes:
        Number of computers in the network.

    connection_probability:
        Probability that two computers are connected.
    """

    network = nx.erdos_renyi_graph(
        number_of_nodes,
        connection_probability,
        seed=42
    )

    return network


# ---------------------------------------------------------
# BASIC NETWORK INFORMATION
# ---------------------------------------------------------

def display_network_info(network):
    """
    Displays basic information about the network.
    """

    print("\n========== NETWORK INFORMATION ==========")

    print("Number of nodes:", network.number_of_nodes())

    print("Number of connections:", network.number_of_edges())

    # Find isolated nodes
    isolated_nodes = list(nx.isolates(network))

    print("Number of isolated nodes:", len(isolated_nodes))

    if len(isolated_nodes) > 0:
        print("Isolated nodes:", isolated_nodes)
    else:
        print("No isolated nodes found.")

    print("=========================================\n")


# ---------------------------------------------------------
# SEQUENTIAL ANALYSIS
# ---------------------------------------------------------

def sequential_analysis(network):
    """
    Calculates clustering coefficient for every node
    using normal sequential processing.
    """

    results = {}

    start_time = time.perf_counter()

    for node in network.nodes():

        clustering_value = nx.clustering(network, node)

        results[node] = clustering_value

    end_time = time.perf_counter()

    execution_time = end_time - start_time

    return results, execution_time


# ---------------------------------------------------------
# INITIALIZE PARALLEL WORKERS
# ---------------------------------------------------------

def initialize_worker(network):
    """
    Gives each worker process access to the network.
    """

    global graph

    graph = network


# ---------------------------------------------------------
# PARALLEL WORKER FUNCTION
# ---------------------------------------------------------

def calculate_node_clustering(node):

    """
    Calculates clustering coefficient for one node.

    This function runs inside a worker process.
    """

    value = nx.clustering(graph, node)

    return node, value


# ---------------------------------------------------------
# PARALLEL ANALYSIS
# ---------------------------------------------------------

def parallel_analysis(network):
    """
    Calculates clustering coefficient for all nodes
    using multiple CPU processes.
    """

    start_time = time.perf_counter()

    # Number of CPU cores
    cpu_count = multiprocessing.cpu_count()

    print("CPU cores available:", cpu_count)

    # Create a pool of worker processes
    with multiprocessing.Pool(
        processes=cpu_count,
        initializer=initialize_worker,
        initargs=(network,)
    ) as pool:

        nodes = list(network.nodes())

        results_list = pool.map(
            calculate_node_clustering,
            nodes
        )

    # Convert list into dictionary
    results = dict(results_list)

    end_time = time.perf_counter()

    execution_time = end_time - start_time

    return results, execution_time


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

def display_analysis_results(results, title):

    print("\n==========", title, "==========")

    # Display only first 10 nodes
    count = 0

    for node, value in results.items():

        print(
            "Node",
            node,
            "-> Clustering coefficient:",
            round(value, 4)
        )

        count += 1

        if count == 10:
            print("... showing first 10 nodes only ...")
            break

    print("=========================================\n")


# ---------------------------------------------------------
# CALCULATE SPEEDUP
# ---------------------------------------------------------

def calculate_speedup(sequential_time, parallel_time):

    if parallel_time == 0:
        return 0

    speedup = sequential_time / parallel_time

    return speedup


# ---------------------------------------------------------
# DRAW NETWORK
# ---------------------------------------------------------

def draw_network(network):

    plt.figure(figsize=(8, 6))

    position = nx.spring_layout(
        network,
        seed=42
    )

    nx.draw(
        network,
        position,
        with_labels=True,
        node_size=500,
        font_size=8
    )

    plt.title("Computer Network Topology")

    plt.show()


# ---------------------------------------------------------
# PERFORMANCE GRAPH
# ---------------------------------------------------------

def draw_performance_graph(
        sequential_time,
        parallel_time
):

    methods = [
        "Sequential",
        "Parallel"
    ]

    times = [
        sequential_time,
        parallel_time
    ]

    plt.figure(figsize=(7, 5))

    plt.bar(methods, times)

    plt.ylabel("Execution Time (seconds)")

    plt.xlabel("Processing Method")

    plt.title("Sequential vs Parallel Performance")

    plt.show()


# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------

def main():

    print("==============================================")
    print("   PARALLEL NETWORK TOPOLOGY ANALYZER")
    print("==============================================")

    # ---------------------------------------------
    # STEP 1: Create network
    # ---------------------------------------------

    number_of_nodes = 100

    network = create_network(
        number_of_nodes,
        connection_probability=0.05
    )

    # ---------------------------------------------
    # STEP 2: Display network information
    # ---------------------------------------------

    display_network_info(network)

    # ---------------------------------------------
    # STEP 3: Display network graph
    # ---------------------------------------------

    draw_network(network)

    # ---------------------------------------------
    # STEP 4: Sequential processing
    # ---------------------------------------------

    print("Running sequential analysis...")

    sequential_results, sequential_time = (
        sequential_analysis(network)
    )

    print(
        "Sequential execution time:",
        round(sequential_time, 6),
        "seconds"
    )

    display_analysis_results(
        sequential_results,
        "SEQUENTIAL RESULTS"
    )

    # ---------------------------------------------
    # STEP 5: Parallel processing
    # ---------------------------------------------

    print("Running parallel analysis...")

    parallel_results, parallel_time = (
        parallel_analysis(network)
    )

    print(
        "Parallel execution time:",
        round(parallel_time, 6),
        "seconds"
    )

    display_analysis_results(
        parallel_results,
        "PARALLEL RESULTS"
    )

    # ---------------------------------------------
    # STEP 6: Calculate speedup
    # ---------------------------------------------

    speedup = calculate_speedup(
        sequential_time,
        parallel_time
    )

    print("\n========== PERFORMANCE ==========")

    print(
        "Sequential time:",
        round(sequential_time, 6),
        "seconds"
    )

    print(
        "Parallel time:",
        round(parallel_time, 6),
        "seconds"
    )

    print(
        "Speedup:",
        round(speedup, 2),
        "x"
    )

    print("=================================\n")

    # ---------------------------------------------
    # STEP 7: Draw performance graph
    # ---------------------------------------------

    draw_performance_graph(
        sequential_time,
        parallel_time
    )


# ---------------------------------------------------------
# PROGRAM ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":

    main()