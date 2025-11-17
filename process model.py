import pm4py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

event_log = pm4py.read_xes("D:\桌面\BPI Challenge 2017_1_all\BPI Challenge 2017.xes")

from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.objects.conversion.bpmn import converter as bpmn_converter
from pm4py.objects.bpmn.obj import BPMN
# Inductive Miner (提供结构化模型)

process_tree_inductive = inductive_miner.apply(event_log)
bpmn_inductive = pm4py.convert_to_bpmn(process_tree_inductive)
pm4py.write_bpmn(bpmn_inductive, "D:/fianfinal_process_model.bpmn")
pm4py.view_bpmn(bpmn_inductive)


# Heuristics Miner (捕获频繁路径)
net_heuristics, initial_marking, final_marking = heuristics_miner.apply(event_log, parameters={
        heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.8,
        heuristics_miner.Variants.CLASSIC.value.Parameters.AND_MEASURE_THRESH: 0.65,
    })
bpmn_heuristics = pm4py.convert_to_bpmn(net_heuristics, initial_marking, final_marking)
pm4py.write_bpmn(bpmn_heuristics, "D:/heuristics_model.bpmn")
pm4py.view_bpmn(bpmn_heuristics)


from pm4py.evaluation.replay_fitness import evaluator as replay_fitness_evaluator
from pm4py.evaluation.precision import evaluator as precision_evaluator
from pm4py.evaluation.generalization import evaluator as generalization_evaluator
from pm4py.evaluation.simplicity import evaluator as simplicity_evaluator
from pm4py.objects.conversion.bpmn import converter as bpmn_converter

def evaluate_model(event_log, bpmn_model, model_name):
    """Evaluate fitness, precision, generalization, and simplicity of a BPMN model."""
    # Convert BPMN to Petri net for evaluation
    net, initial_marking, final_marking = bpmn_converter.apply(bpmn_model)
    
    # Calculate metrics
    fitness = replay_fitness_evaluator.apply(
        event_log, net, initial_marking, final_marking,
        variant=replay_fitness_evaluator.Variants.TOKEN_BASED
    )
    precision = precision_evaluator.apply(
        event_log, net, initial_marking, final_marking,
        variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN
    )
    generalization = generalization_evaluator.apply(
        event_log, net, initial_marking, final_marking
    )
    simplicity = simplicity_evaluator.apply(net)
    
    # Print results
    print(f"\n--- {model_name} Evaluation ---")
    print(f"Fitness (Avg Trace Fitness): {fitness['average_trace_fitness']:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Generalization: {generalization:.4f}")
    print(f"Simplicity: {simplicity:.4f}")
    
    return {
        "fitness": fitness['average_trace_fitness'],
        "precision": precision,
        "generalization": generalization,
        "simplicity": simplicity
    }

# Evaluate both initial models
inductive_results = evaluate_model(event_log, bpmn_inductive, "Inductive Miner Model")
heuristics_results = evaluate_model(event_log, bpmn_heuristics, "Heuristics Miner Model")

import graphviz

# BPMN
def create_simplified_bpmn():
    bpmn_graph = BPMN()
    
    start_event = BPMN.StartEvent(name="Start Application")
    bpmn_graph.add_node(start_event)
    
    tasks = [
        "W_Complete application",
        "W_Validate application", 
        "W_Assess potential fraud",
        "W_Call after offers",
        "W_Call incomplete files"
    ]
    
    previous_node = start_event
    for task in tasks:
        task_node = BPMN.Task(name=task)
        bpmn_graph.add_node(task_node)
        flow = BPMN.SequenceFlow(previous_node, task_node)
        bpmn_graph.add_flow(flow)
        previous_node = task_node
    
    end_event = BPMN.EndEvent(name="Application Completed")
    bpmn_graph.add_node(end_event)
    flow = BPMN.SequenceFlow(previous_node, end_event)
    bpmn_graph.add_flow(flow)
    
    return bpmn_graph

simple_bpmn = create_simplified_bpmn()
pm4py.write_bpmn(simple_bpmn, "D:/simplified_model.bpmn")
BPMN_results = evaluate_model(event_log, simple_bpmn, "Simple BPMN Model")