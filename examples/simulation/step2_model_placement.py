# 2024.10.29 Yixuan Mei
import sys
from simulator.initial_layout.layout_synthesizer import LayoutMethod, LayoutSynthesizer, ModelName
from simulator.event_simulator.utils import kbps, mbps, gbps, KB, MB, GB, Sec, MilliSec


def petals_layout():
    # heuristic method: petals
    layout_synthesizer = LayoutSynthesizer(
        complete_cluster_file_name="./config/single24.ini",
        machine_profile_name="./config/machine_profile.ini",
        model_name=ModelName.LLaMa70B,
        workspace_path="./layouts/petals",
        layout_method=LayoutMethod.Petals,
        machine_num_dict={"A100": 4, "L4": 8, "T4": 12}
    )
    petals_args = {
        "seed": 0,
        "max_out_links_per_node": 24,
    }
    layout_synthesizer.synthesize(args=petals_args)


def swarm_layout():
    # heuristic method: swarm
    layout_synthesizer = LayoutSynthesizer(
        complete_cluster_file_name="./config/single24.ini",
        machine_profile_name="./config/machine_profile.ini",
        model_name=ModelName.LLaMa70B,
        workspace_path="./layouts/swarm",
        layout_method=LayoutMethod.Swarm,
        machine_num_dict={"A100": 4, "L4": 8, "T4": 12}
    )
    swarm_args = {
        "seed": 0,
        "num_stages": 20,  # as few as possible
        "max_out_links_per_node": 24,
    }
    layout_synthesizer.synthesize(args=swarm_args)


def homogeneous_layout():
    # heuristic method: homogeneous
    layout_synthesizer = LayoutSynthesizer(
        complete_cluster_file_name="./config/single24.ini",
        machine_profile_name="./config/machine_profile.ini",
        model_name=ModelName.LLaMa70B,
        workspace_path="./layouts/homogeneous",
        layout_method=LayoutMethod.Homogeneous,
        machine_num_dict={"A100": 4, "L4": 8, "T4": 12}
    )
    homogeneous_args = {
        "seed": 0,
    }
    layout_synthesizer.synthesize(args=homogeneous_args)


def ilp_layout():
    # initialize the layout synthesizer
    layout_synthesizer = LayoutSynthesizer(
        complete_cluster_file_name="./config/single24.ini",
        machine_profile_name="./config/machine_profile.ini",
        model_name=ModelName.LLaMa70B,
        workspace_path="./layouts/ilp",
        layout_method=LayoutMethod.ILP,
        machine_num_dict={"A100": 4, "L4": 8, "T4": 12}
    )

    # setting arguments for ILP layout synthesis
    # see simulator.initial_layout.layout_synthesizer.synthesize for more details about the arguments
    ilp_args = {
        # pruning
        # pruning removes some edges in the graph to reduce the problem size
        "enable_pruning": False,
        "min_keep": 12,
        "max_keep": 12,
        "keep_bandwidth_threshold": 1 * mbps,
        # ILP
        # if "use_existing_sol" is True, the synthesizer will only load an existing solution and verify it
        # if you want to continue optimize from an existing solution, you can use "start_from_heuristic" below
        # here, if "use_existing_sol" is True, the "existing_sol_path" should be the ilp_solution.sol you
        # want to verify
        "use_existing_sol": False,
        "allow_partial_inference": False,
        "remove_redundant": True,
        "max_run_time": 36000,
        "early_stop_time": 100,
        "early_stop_threshold": 0.95,
        "existing_sol_path": "path/to/existing/ilp_solution.sol",
        # heuristic
        # if this is true, then the MILP solver will load an existing solution (generated by some
        # heuristic methods or the ILP layout itself) and continue optimize from there
        "start_from_heuristic": True,
        "heuristic_sol_path": "./layouts/petals/petals_sol.ini",
    }

    # run the ILP layout synthesis
    layout_synthesizer.synthesize(args=ilp_args)


def main():
    """
    The second step is to find a model placement for the cluster. The model placement specifies which
    layers each machine holds.
    Helix simulator supports four layout synthesis methods:
     1. ILP: the MILP-based layout method in Helix
     2. Petals: our implementation of Petals' layout method
     3. Swarm: our implementation of Swarm's layout method
     4. Homogeneous: similar to Orca, each pipeline contains the same type of machines
    """
    assert len(sys.argv) == 2, f"Usage: python {sys.argv[0]} <layout_method> (ilp/swarm/petals/homogeneous)"
    layout_method = sys.argv[1]

    if layout_method == "ilp":
        # ILP layout synthesis
        # Note: We set the max running time to 10 hours. However, you can stop the process at any time (ctrl + c ONCE)
        # and the best solution found so far will be saved. In this example, we early stop at around 10 minutes.
        # Depending on the random seed, the running time and model placement found may vary.
        ilp_layout()
        print(f"ILP layout synthesis is done! (Results in ./layouts/ilp)")

    elif layout_method == "swarm":
        # Heuristic method: Swarm
        swarm_layout()
        print(f"Swarm layout synthesis is done! (Results in ./layouts/swarm)")

    elif layout_method == "petals":
        # Heuristic method: Petals
        petals_layout()
        print(f"Petals layout synthesis is done! (Results in ./layouts/petals)")

    elif layout_method == "homogeneous":
        # Heuristic method: Homogeneous
        homogeneous_layout()
        print(f"Homogeneous layout synthesis is done! (Results in ./layouts/homogeneous)")

    else:
        raise ValueError(f"Unknown layout method: {layout_method}")


if __name__ == '__main__':
    main()
