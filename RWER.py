'''
This module implements a variant of a random walk on a directed graph. In this
variant, a restart set consisting of edges, rather than nodes, is provided. A
random walker may restart to the middle of one of these edges as it transitions
along the graph.
'''

import sys
import itertools

from typing import Dict, List, NamedTuple, NewType
from collections import namedtuple

import networkx as nx
import networkx.DiGraph as DiGraph 

import pathlinker.PathLinker as pl
import pathlinker.PageRank as pr 

Node = NewType('Node', int)

class Edge(NamedTuple):
    tail: Node 
    head: Node 


def __get_new_mapping(graph: DiGraph):
    '''
    Return a mapping of original node labels to non-negative integers
    '''
    mapping = {}

    for node, i in zip(graph.nodes(), itertools.count(0)):
        mapping[node] = Node(i)

    return mapping


def __map_edges_to_ids(edges: List[Edge], start) -> Dict[Edge, Node]:
    '''
    :param edges: list of edges to map
    :param start: integer ID to start with
    '''
    mapping = Dict[Edge, Node]

    for edge, i in zip(edges, itertools.count(start)):
        mapping[edge] = Node(i)

    return mapping

# TODO: "intermediate" is not really the right term
def __add_intermediate_nodes_and_edges(
        graph: DiGraph, restart_set: List[Edge]):
    '''
    For every edge, add a corresponding node
    '''

    mapping = __map_edges_to_ids(restart_set, len(graph.nodes()))

    for e in restart_set:
        graph.add_edge(
            mapping[e], e.head, attr_dict=graph.get_edge_data(e.tail, e.head))

    return mapping 


def __set_minimum_edge_weight(graph, weight_name):
    '''
    Ensure that every edge has a non-zero weight
    '''
    for edge in graph.edges(data=True):
        if edge[2][weight_name] == 0:
            edge[2][weight_name] = sys.float_info.min


def rwer(graph: DiGraph, restart_set: List[Edge], q: float, eps: float, 
        max_iters: int, weight_name='weight', verbose=False):
    '''
    Performs RWER and returns the flux across each edge.

    :param graph: NetworkX DiGraph on which to perform RWER
    :param restart_set: a list of of edges to restart to
    :param q: restart probability
    :param eps: PageRank convergence threshold
    :param max_iters: maximum number of PageRank iterations
    :param weight_name: name of weight field for edges
    '''

    # Map node labels to integers for simplicity.
    mapping = __get_new_mapping(graph)

    # Copy the graph so that any modifications we perform are temporary.
    graph_copy = nx.relabel_graph(graph, mapping)

    # Add intermediate nodes and edges for every edge in the restart set
    edge_mapping = __add_intermediate_nodes_and_edges(graph_copy, restart_set)

    # Give all new nodes a weight of 1
    weights = {node:1 for node in edge_mapping.values()}

    __set_minimum_edge_weight(graph_copy, weight_name)

    pagerank_scores = pr.pagerank(
        graph_copy, weights=weights, q=q, eps=eps, maxIters=max_iters, 
        weightName=weight_name, verbose=verbose)

    pl.calculateFluxEdgeWeights(graph_copy, pagerank_scores)

    # Now we calculate the true flux
    true_fluxes: Dict[Edge, float] = {}

    # First, loop over all the original edges and get their corresponding flux
    for tail, head in graph.edges():
        tail = mapping(tail)
        head = mapping(head)

        true_fluxes[Edge(tail, head)] = \
            graph_copy.get_edge_data(tail, head)['ksp_weight']
    
    # Next, add the flux from the intermediate edges to the flux of edges in
    # the restart set. If we have edge 3->4 in the restart set, we add edge
    # 5->4 We need to get the flux from 5->4 and add it to 3->4
    for edge in edge_mapping.keys():
        tail = edge_mapping[edge]
        head = edge.head

        additional_flux = graph_copy.get_edge_data(tail, head)['ksp_weight']

        true_fluxes[edge] += additional_flux

    # Finally, map the edge names back to their original names 
    true_fluxes_remapped = {}

    for edge in true_fluxes.keys():
        original_tail = mapping(edge.tail)
        original_head = mapping(edge.head)

        true_fluxes_remapped[(original_tail, original_head)] = \
            true_fluxes(edge)

    return true_fluxes_remapped
