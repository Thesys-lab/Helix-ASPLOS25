# 2023.03.12 Yixuan Mei

import math
import random
from configparser import ConfigParser
from typing import Dict, Tuple, List

from simulator.event_simulator.utils import kbps, mbps, gbps, Byte, KB, MB, GB, Sec, MilliSec
from simulator.event_simulator.cluster_simulator import ClusterSimulator
from simulator.event_simulator.query_manager import QueryManagerParameters
from simulator.initial_layout.ilp_layout.ilp_layout import MachineProfile, ModelCard, ILPNode, ILPLink
from simulator.model_manager.model_manager import ModelManager
from simulator.scheduler.global_maxflow.global_maxflow_scheduler import FlowParameters


class HomogeneousLayout:
    def __init__(self, model_manager: ModelManager) -> None:
        """
        Homogeneous layout. (similar to Orca)

        :return: None
        """
        # loaded problem information
        self.machine_profiles: Dict[str, MachineProfile] = {}
        self.model_card: ModelCard or None = None
        self.model_manager: ModelManager = model_manager

        # topology information
        self.source: ILPNode or None = None
        self.sink: ILPNode or None = None
        self.nodes: Dict[int, ILPNode] = {}
        self.links: Dict[Tuple[int or str, int or str], ILPLink] = {}
        self.cluster_loaded: bool = False
        self.solution_found: bool = False

        # pipelines
        self.pipelines: List[List[ILPNode]] = []
        self.nodes_in_use: Dict[int, ILPNode] = {}
        self.node_idx_offset: int = 2  # for cluster file generation

    def from_ini(self, cluster_file_name: str, machine_profile_name: str) -> None:
        """
        Load cluster topology and machine profiles.

        :param cluster_file_name: name of the file that stores cluster topology
        :param machine_profile_name: name of the file that stores machine profiling results
        :return: None
        """
        # clear the dicts
        self.machine_profiles.clear()
        self.nodes.clear()
        self.links.clear()

        # load machine statistics
        machine_profile_parser = ConfigParser()
        machine_profile_parser.read(machine_profile_name)
        for machine_name in machine_profile_parser.sections():
            self.machine_profiles[machine_name] = MachineProfile(machine_name=machine_name,
                                                                 config=machine_profile_parser)

        # load cluster topology
        cluster_file_parser = ConfigParser()
        cluster_file_parser.read(cluster_file_name)

        # check that the topology is a complete graph
        total_num_compute_nodes = eval(cluster_file_parser["NodeNames"]["total_compute_nodes"])
        assert sorted(eval(cluster_file_parser["SourceNode"]["connected_nodes"])) == list(
            range(total_num_compute_nodes)), "Not a complete graph!"
        assert sorted(eval(cluster_file_parser["SinkNode"]["connected_nodes"])) == list(
            range(total_num_compute_nodes)), "Not a complete graph!"
        for i in range(total_num_compute_nodes):
            connected_nodes: List[int or str] = ["source"] + list(range(total_num_compute_nodes)) + ["sink"]
            connected_nodes.remove(i)
            assert eval(cluster_file_parser[f"ComputeNode-{i}"]["connected_nodes"]) == connected_nodes, \
                "Not a complete graph!"

        # model
        self.model_card = ModelCard(model_manager=self.model_manager)

        # source and sink
        self.source = ILPNode(node_index=-1, machine_type=self.machine_profiles["SourceNode"], max_num_layers=-1,
                              connected_node_indices=eval(cluster_file_parser["SourceNode"]["connected_nodes"]),
                              layer_count_2_throughput={})
        self.sink = ILPNode(node_index=-1, machine_type=self.machine_profiles["SinkNode"], max_num_layers=-1,
                            connected_node_indices=eval(cluster_file_parser["SinkNode"]["connected_nodes"]),
                            layer_count_2_throughput={})

        # compute nodes
        total_compute_nodes: int = eval(cluster_file_parser["NodeNames"]["total_compute_nodes"])
        for node_idx in range(total_compute_nodes):
            # extract machine name, type and connected nodes from file
            machine_name: str = f"ComputeNode-{node_idx}"
            machine_type: MachineProfile = self.machine_profiles[cluster_file_parser[machine_name]["type"]]
            connected_nodes: List[int] = eval(cluster_file_parser[machine_name]["connected_nodes"])

            # compute max number of layers that can be stored on this node
            # Note: max # layers = (VRAM size / 2) / layer size
            max_num_layers: int = self.model_manager.get_max_num_layers(machine_type=machine_type.type_name)
            assert 2 * max_num_layers * max(self.model_manager.get_model_params()) <= machine_type.vram_size + 1, \
                "Trying to use more than half the vram to load model parameters!"

            # compute layer count to throughput
            # Note: 1. inference throughput is computed under typical batch size
            #       2. total throughput is the min of inference throughput and nic throughput
            bottleneck_nic_speed: float = min(machine_type.inbound_nic_speed, machine_type.outbound_nic_speed)
            bottleneck_nic_throughput: float = bottleneck_nic_speed / self.model_card.activation_size
            layer_count_2_throughput: Dict[int, float] = {}
            for layer_count in range(1, max_num_layers + 1):
                inference_throughput: float = self.model_manager.get_typical_token_throughput(
                    machine_type=machine_type.type_name, num_on_node_layers=layer_count
                )
                layer_count_2_throughput[layer_count] = min(inference_throughput, bottleneck_nic_throughput)

            # add node
            self.nodes[node_idx] = ILPNode(node_index=node_idx,
                                           machine_type=machine_type,
                                           max_num_layers=max_num_layers,
                                           connected_node_indices=connected_nodes,
                                           layer_count_2_throughput=layer_count_2_throughput)

        # links
        # Note: links here are bidirectional
        for entity_name in cluster_file_parser.sections():
            if "Link-" in entity_name:
                # end points
                from_idx: int or str = entity_name.split("-")[1]
                if not from_idx == "source":
                    from_idx = int(from_idx)
                to_idx: int or str = entity_name.split("-")[2]
                if not to_idx == "sink":
                    to_idx = int(to_idx)

                # bandwidth and latency
                bandwidth: float = eval(cluster_file_parser[entity_name]["bandwidth"])
                latency: float = eval(cluster_file_parser[entity_name]["latency"])
                if from_idx == "source" or to_idx == "sink":
                    throughput: float = bandwidth / self.model_card.token_size
                else:
                    assert isinstance(from_idx, int) and isinstance(to_idx, int), "Bad index!"
                    throughput: float = bandwidth / self.model_card.activation_size
                self.links[(from_idx, to_idx)] = ILPLink(from_index=from_idx,
                                                         to_index=to_idx,
                                                         throughput=throughput,
                                                         bandwidth=bandwidth,
                                                         latency=latency)

        # mark cluster as loaded
        self.cluster_loaded = True

    def synthesize(self, seed: int) -> None:
        """
        Synthesize an initial model layout.
        Note: 1. Each pipeline is a sequence of machines with the same type. Pipelines are independent.
              2. If machine from one type is not enough to form a pipeline, we will use a machine from another
                 type to fill the gap.

        :param seed: random seed
        :return: None
        """
        random.seed(seed)
        assert self.cluster_loaded, "Cluster must be loaded before we can do initial layout synthesis!"

        # determine the performance ordering of each type of machine
        machine_type_perf: List[Tuple[float, str]] = []
        for machine_type in self.model_manager.machine_num_dict.keys():
            if machine_type == "SourceNode" or machine_type == "SinkNode":
                continue
            max_layers_on_node = self.model_manager.get_max_num_layers(machine_type=machine_type)
            typical_throughput = self.model_manager.get_typical_token_throughput(
                machine_type=machine_type, num_on_node_layers=max_layers_on_node
            )
            machine_type_perf.append((typical_throughput * max_layers_on_node, machine_type))
        machine_type_perf.sort(reverse=True)

        # categorize the machines by their type
        type2machine: Dict[str, List[ILPNode]] = {type_perf[1]: [] for type_perf in machine_type_perf}
        for compute_node in self.nodes.values():
            type2machine[compute_node.machine_type.type_name].append(compute_node)

        # randomize the order of the machines
        for machines in type2machine.values():
            random.shuffle(machines)

        # form the pipelines
        pipelines: List[List[ILPNode]] = []
        nodes_in_use: Dict[int, ILPNode] = {}
        total_num_layers: int = self.model_card.num_layers
        cur_pipeline: List[ILPNode] = []
        cur_pipeline_length = 0
        for _, type_name in machine_type_perf:
            for machine in type2machine[type_name]:
                # put current machine into pipeline
                machine.start_layer_idx = cur_pipeline_length
                machine.end_layer_idx = min(cur_pipeline_length + machine.max_num_layers, total_num_layers)
                cur_pipeline.append(machine)

                # proceed to next
                cur_pipeline_length = machine.end_layer_idx
                if cur_pipeline_length == total_num_layers:
                    pipelines.append(cur_pipeline)
                    for node in cur_pipeline:
                        nodes_in_use[node.node_index] = node
                    cur_pipeline = []
                    cur_pipeline_length = 0

        # check each pipeline is complete and store the pipelines
        for pipeline in pipelines:
            assert pipeline[-1].end_layer_idx == total_num_layers, "Found incomplete pipeline!"
        self.pipelines = pipelines
        self.nodes_in_use = nodes_in_use
        self.solution_found = True

    def generate_simulator_cluster(self, cluster_file_path: str) -> None:
        """
        Generate cluster file and statistics file for simulation.

        :param cluster_file_path: path to cluster file
        :return: None
        """
        assert self.solution_found, "Must run synthesis before generating cluster files!"

        # generate cluster file
        with open(cluster_file_path, "w") as file:
            # header notes
            file.write("# Simulator cluster file generated by Homogeneous layout synthesizer.\n")
            file.write("\n")

            # write coordinator
            file.write(f"[Coordinator]\n")
            inbound_nic_speed: float = self.sink.machine_type.inbound_nic_speed / mbps
            outbound_nic_speed: float = self.source.machine_type.outbound_nic_speed / mbps
            file.write(f"inbound_nic_speed={inbound_nic_speed} * mbps\n")
            file.write(f"outbound_nic_speed={outbound_nic_speed} * mbps\n")
            file.write("\n")

            # write machine types
            file.write(f"[MachineTypes]\n")
            machine_types = list(self.machine_profiles.keys())
            machine_types.remove("SourceNode")
            machine_types.remove("SinkNode")
            file.write(f"types={machine_types}\n")
            file.write("\n")

            # write node names
            file.write("[ComputeNodes]\n")
            nodes_in_use_idx: List[int] = sorted(list(self.nodes_in_use.keys()))
            node_names = [f"compute_node_{self.node_idx_offset + idx}" for idx in nodes_in_use_idx]
            file.write(f"names={node_names}\n")
            file.write("\n")

            # write the nodes
            for node_idx, compute_node in self.nodes.items():
                if node_idx not in self.nodes_in_use:
                    continue
                file.write(f"[compute_node_{self.node_idx_offset + node_idx}]\n")
                vram_size: float = compute_node.machine_type.vram_size / MB
                file.write(f"vram_size={vram_size} * MB\n")
                inbound_nic_speed: float = compute_node.machine_type.inbound_nic_speed / mbps
                file.write(f"inbound_nic_speed={inbound_nic_speed} * mbps\n")
                outbound_nic_speed: float = compute_node.machine_type.outbound_nic_speed / mbps
                file.write(f"outbound_nic_speed={outbound_nic_speed} * mbps\n")
                disk_speed: float = compute_node.machine_type.disk_speed / mbps
                file.write(f"disk_speed={disk_speed} * mbps\n")
                file.write(f"machine_type=\"{compute_node.machine_type.type_name}\"\n")
                kv_cache_capacity: int = self.model_manager.get_kv_cache_capacity(
                    machine_type=compute_node.machine_type.type_name,
                    num_on_node_layers=compute_node.end_layer_idx - compute_node.start_layer_idx
                )
                file.write(f"kv_cache_capacity={kv_cache_capacity}\n")
                activation_backup_capacity: int = self.model_manager.get_activation_backup_capacity(
                    machine_type=compute_node.machine_type.type_name,
                    num_on_node_layers=compute_node.end_layer_idx - compute_node.start_layer_idx
                )
                file.write(f"activation_backup_capacity={activation_backup_capacity}\n")
                file.write("\n")

            # write the links
            valid_links: Dict[Tuple[int or str, int or str], ILPLink] = {}
            # source -> pipeline
            for pipeline in self.pipelines:
                first_node_idx = pipeline[0].node_index
                valid_links[("source", first_node_idx)] = self.links[("source", first_node_idx)]
            # pipeline -> sink
            for pipeline in self.pipelines:
                last_node_idx = pipeline[-1].node_index
                valid_links[(last_node_idx, "sink")] = self.links[(last_node_idx, "sink")]
            # in the pipelines
            for pipeline in self.pipelines:
                for i in range(len(pipeline) - 1):
                    from_idx = pipeline[i].node_index
                    to_idx = pipeline[i + 1].node_index
                    if (from_idx, to_idx) in self.links:
                        valid_links[(from_idx, to_idx)] = self.links[(from_idx, to_idx)]
                    else:
                        valid_links[(from_idx, to_idx)] = self.links[(to_idx, from_idx)]

            # write the valid link names
            file.write("[Links]\n")
            valid_link_names = []
            for valid_link_name_tuple in valid_links.keys():
                from_name = valid_link_name_tuple[0] if valid_link_name_tuple[0] == "source" else \
                    f"compute_node_{self.node_idx_offset + valid_link_name_tuple[0]}"
                to_name = valid_link_name_tuple[1] if valid_link_name_tuple[1] == "sink" else \
                    f"compute_node_{self.node_idx_offset + valid_link_name_tuple[1]}"
                valid_link_names.append(f"link_{from_name}_{to_name}")
            file.write(f"names={valid_link_names}\n")
            file.write("\n")

            # write the links
            for valid_link_name_tuple, valid_link in valid_links.items():
                from_name = valid_link_name_tuple[0] if valid_link_name_tuple[0] == "source" else \
                    f"compute_node_{self.node_idx_offset + valid_link_name_tuple[0]}"
                to_name = valid_link_name_tuple[1] if valid_link_name_tuple[1] == "sink" else \
                    f"compute_node_{self.node_idx_offset + valid_link_name_tuple[1]}"
                file.write(f"[link_{from_name}_{to_name}]\n")
                file.write(f"in={from_name}\n")
                file.write(f"out={to_name}\n")
                file.write(f"latency={valid_link.latency * 1000} * MilliSec\n")
                file.write(f"bandwidth={valid_link.bandwidth / mbps} * mbps\n")
                file.write("\n")

    def save_layout_solution(self, save_path: str) -> None:
        """
        Save the layout solution found.
        Format:
        [Solution]
        name_in_cluster_file=[a list of layer ids]

        :param save_path: save path of solution file
        :return: None
        """
        assert self.solution_found, "Must find a solution before saving!"
        with open(save_path, "w") as file:
            file.write("[Settings]\n")
            file.write(f"offset={self.node_idx_offset}\n")
            file.write("\n")
            file.write("[Solution]\n")
            for ilp_node_idx, ilp_node in self.nodes.items():
                file.write(f"compute_node_{self.node_idx_offset + ilp_node_idx}=")
                if ilp_node_idx in self.nodes_in_use:
                    file.write(f"{list(range(ilp_node.start_layer_idx, ilp_node.end_layer_idx))}\n")
                else:
                    file.write("[]\n")

    def set_initial_layout(self, simulator: ClusterSimulator) -> float:
        """
        Load the initial model layout into the simulator.

        :param simulator: the cluster simulator to load model into
        :return: expected loading time in simulation
        """
        assert self.solution_found, "Must synthesize a solution before setting initial layout for simulator!"
        assert simulator.current_time == 0, "Initial layout can only be set at the beginning!"

        max_load_time: float = 0
        for ilp_node_idx, ilp_node in self.nodes_in_use.items():
            # get the corresponding compute node in the simulator
            compute_node_name = f"compute_node_{self.node_idx_offset + ilp_node_idx}"
            compute_node = simulator.name_2_compute_node[compute_node_name]

            # get the model layers to load and corresponding loading time
            new_layers = list(range(ilp_node.start_layer_idx, ilp_node.end_layer_idx))
            new_layers_size = sum(self.model_manager.get_model_params()[ilp_node.start_layer_idx:
                                                                        ilp_node.end_layer_idx])
            loading_time = new_layers_size / compute_node.disk_speed
            max_load_time = max(max_load_time, loading_time)

            # issue load command
            simulator.issue_command_load_model(load_time=simulator.current_time,
                                               node_uid=compute_node.node_uid,
                                               new_layers=new_layers,
                                               request_uids_to_wait=[])

        # advance simulator
        max_load_time = math.ceil(max_load_time) + 1
        simulator.simulate(until=max_load_time)
        return max_load_time

    def get_flow_upper_bound(self) -> float:
        """
        Get the upper bound of max flow over this cluster, which is defined as the max flow when all network
        transmissions are instant.

        :return: flow upper bound
        """
        assert self.cluster_loaded, "Cluster must be loaded before we can compute flow upper bound!"
        total_compute_throughput: float = 0
        for node_idx, compute_node in self.nodes.items():
            cur_node_max = -1
            for i in range(1, compute_node.max_num_layers + 1):
                cur_node_max = max(cur_node_max, compute_node.layer_count_2_throughput[i] * i)
            assert not cur_node_max == -1, "Bad max throughput!"
            total_compute_throughput += cur_node_max
        return total_compute_throughput / self.model_card.num_layers

    def get_flow_parameters(self) -> FlowParameters:
        """
        Get flow parameters based on the loaded cluster file.

        :return: FlowParameters
        """
        assert self.solution_found, "Solution must be loaded before FlowParameters can be returned!"
        return FlowParameters(token_size=self.model_card.token_size,
                              token_activation_size=self.model_card.activation_size)

    def get_query_manager_parameters(self) -> QueryManagerParameters:
        """
        Get query manager parameters based on the loaded cluster file.

        :return: QueryManagerParameters
        """
        assert self.solution_found, "Solution must be loaded before QueryManagerParameters can be returned!"
        return QueryManagerParameters(token_size=self.model_card.token_size,
                                      token_activation_size=self.model_card.activation_size,
                                      total_num_layers=self.model_card.num_layers)
