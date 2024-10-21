import os
import logging
from random import randint
import random
import errno
import functools
from loguru import logger
import signal
import time
import ray
import json
import faiss
import subprocess
import shutil
import psutil
import pickle
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

import lean_dojo
from lean_dojo import *
from utils.lean_math_utils import *

#from memory_profiler import profile

from lean_dojo.interaction.dojo import (
    LeanError,
    TimeoutError,
    TacticResult,
    DojoCrashError,
    DojoHardTimeoutError,
    DojoInitError,
    ProofGivenUp,
    ProofFinished,
)


MAX_MEMORY_USAGE = 16*1024*1024*1024
MAX_STEPS = 100000
MAX_TACTIC_FROM_TEMPLATE = 50
PENALTY_SEEN_TARGET_MULTIPLIER = 3
MAX_NUM_DOJO_ATTEMPT = 2
MAX_NUM_OUTPUT_PER_STATE = 50

class TripletDataset(Dataset):
    def __init__(self, triplets, tokenizer):
        self.triplets = triplets
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        q, s, n = self.triplets[index]
        q_encoded = self.tokenizer(q, padding='max_length', max_length=256, truncation=True, return_tensors='pt')
        s_encoded = self.tokenizer(s, padding='max_length', max_length=64, truncation=True, return_tensors='pt')
        n_encoded = self.tokenizer(n, padding='max_length', max_length=64, truncation=True, return_tensors='pt')
        return q_encoded, s_encoded, n_encoded

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, q_embed, s_embed, n_embed):
        dist_pos = torch.norm(q_embed - s_embed, p=2, dim=1)
        dist_neg = torch.norm(q_embed - n_embed, p=2, dim=1)
        loss = torch.mean(torch.relu(dist_pos - dist_neg + self.margin))
        return loss
    
# Example query string

def get_similar_tacs(query_string, model_state, tokenizer, faiss_index, all_tacs, num_returned=100, verbose=False):
    # Convert query string to BERT embedding
    device = model_state.device
    with torch.no_grad():
        encoded_input = tokenizer(query_string, padding=True, truncation=True, return_tensors='pt').to(device)
        outputs = model_state(**encoded_input)
        query_embedding = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()

    # Perform nearest neighbor search
    distances, indices = faiss_index.search(query_embedding, num_returned)

    if verbose:
        # Print the nearest neighbors and their distances
        print(f"Nearest neighbors for: {query_string}")
    output = []
    for i in range(num_returned):
        idx = indices[0][i]
        distance = distances[0][i]
        similarity = 1.0 / (1 + distance*distance/1000)
        tac = all_tacs[idx]
        output.append((all_tacs[idx],similarity))
        if verbose:
            print(f"{i+1}. {all_tacs[idx]} (similarity: {similarity})")
    return sorted(output, key=lambda item: item[1], reverse=True)


# The complexity of a state if calculated from lengths of its targets
# seen_target_freq is a dict{string, int} of strings containing lines of targets that have been seen
# If all targets in the current state has been seen, add to complexity PENALTY_SEEN_TARGET_MULTIPLIER*min_freq
# At the same time, seen_target_freq is updated by the targets in this state
# if base_complexity is not none, complexity is base_complexity+1
def explore_state_complexity(state, base_complexity=None, seen_target_freq=None):
    if '⊢ False' in state.pp:
        return 1000000
    if base_complexity is not None:
        return base_complexity + 1
    complexity = 0
    lines = state.pp.split('\n')
    targets = []
    min_freq = 1000000
    for line in lines:
        if line.startswith("⊢"):
            target = line[1:].strip()
            targets.append(target)
            if seen_target_freq is not None:
                if target in seen_target_freq:
                    min_freq = min(min_freq, seen_target_freq[target])
                    seen_target_freq[target] = seen_target_freq[target] + 1
                else:
                    min_freq = 0
                    seen_target_freq[target] = 1
    lengths = [len(x) for x in targets]
    complexity = max(lengths)/2 + sum(lengths)/2
    if seen_target_freq is not None:
        complexity += PENALTY_SEEN_TARGET_MULTIPLIER*min_freq
    return complexity

# Print the trace from state_0 to state
def get_tactic_trace(curr_state, state_dict):
    _, parent_states, tactics = state_dict[curr_state]
    if len(parent_states) == 0:
        return []
    else:
        return get_tactic_trace(parent_states[0], state_dict) + [(tactics[0], curr_state)]
    
# Create the inverse tactic for a 'rw' tactic template
def get_inverse_tactic(tac_template):
    if not tac_template.startswith('rw') or not '[' in tac_template \
        or '←' in tac_template or '[{' in tac_template:
        return None
    result = tac_template.replace('[', '[← ')
    if ',' in result:
        pos = result.find(',')
        result = result[0:pos] + ']'
    return result

@timeout(1)
def dojo_run_tac(dojo, state, tactic):
    #     # Get the current process object
    #     process = psutil.Process(os.getpid())
    #     # Memory info (in bytes)
    #     mem_info = process.memory_info()
    #     # Print the RSS (Resident Set Size): total physical memory used
    #     print(f"Current memory RSS usage: {mem_info.rss / (1024 ** 2):.2f} MB", randint(0,100), end="")
    #print("Run tac", end="")
    result = dojo.run_tac(state, tactic)
    #print("Done")
    return result

#@profile
def explore_states(dojo,
                   state_0,
                   theorem,
                   model_state,
                   tokenizer,
                   rag_index,
                   all_tacs, 
                   theorem_code,
                   proof_tactics=None,
                   max_steps=MAX_STEPS, 
                   max_time=1200, 
                   exit_on_finish=True,
                   verbose=False):
    start_time = time.time()

    #if verbose: print(state_0.pp)
    unused_vars = []
    if theorem_code is not None:
        tokens = tokenize_lean_tactic(theorem_code)
        tokens = [x for x in tokens if x.strip() != '']
        type_of_item, def_of_item = classify_lean_elements(state_0.pp)
        unused_vars = [x for x in type_of_item.keys() if x not in tokens]
        #print(unused_vars)

    state_queue = PriorityQueue() # PQ of states, priority being complexity of the state
    state_dict = {} # Keyed by state.pp (or state if it is ProofFinished). Value is (state, its parent states, and tactics from parent state to this state)

    curr_state = state_0
    base_complexity = 0
    state_queue.push(curr_state, explore_state_complexity(curr_state, base_complexity=base_complexity) + randint(0, 4))
    base_complexity += 1
    
    # State graph: Key is state.pp (or ProofFinished state). Value is (state, [parent states], [tactic for each parent state], path from state_0 to state)
    state_dict[curr_state.pp] = (curr_state, list(), list(), list())
    
    # Follow traced_tactics and populate state_dict with the authoratative proof
    if proof_tactics is not None:
        for tactic in proof_tactics:
            try:
                test_state = dojo_run_tac(dojo, curr_state, tactic)
            except:
                break
            if type(test_state) in [LeanError,TimeoutError,TacticResult,DojoCrashError,DojoHardTimeoutError,DojoInitError,ProofGivenUp]:
                break
            elif type(test_state) == lean_dojo.interaction.dojo.ProofFinished:
                state_dict[test_state] = (test_state, [curr_state], [tactic], state_dict[curr_state.pp][3] + [tactic])
                print("Successfully followed proof")
                break
            else:
                state_dict[test_state.pp] = (test_state, [curr_state], [tactic], state_dict[curr_state.pp][3] + [tactic]) #store shortest path from state_0
                curr_state = test_state
    n_steps = 0
    theorem_proven = False
    while state_queue.size() > 0:
        curr_state = state_queue.pop()
        if not hasattr(curr_state, 'pp'):
            continue
        #print("POP STATE: ", base_complexity, curr_state.pp, "\nEND STATE")
        type_of_item, def_of_item = classify_lean_elements(curr_state.pp)
        for var in unused_vars:
            if var in type_of_item:
                del type_of_item[var]
            if var in def_of_item:
                del def_of_item[var]
        if verbose:
            if n_steps == 0:
                print("INIT_STATE:", curr_state.pp)
                print(type_of_item)
        #
        #TODO: Add {'nvar0': 'nvar0', 'nvar1': 'nvar1', ...} to type_of_item
        #
        tactic_set = set()
        query = theorem_code + ' # ' + curr_state.pp
        suggestions = get_similar_tacs(query, model_state, tokenizer, rag_index, all_tacs, num_returned=200)
        tac_templates = [x[0] for x in suggestions]
        #
        # Add inverse tactics of any 'rw' tactics in the set
        inv_tac_templates = []
        for tac_template in tac_templates:
            inv_template = get_inverse_tactic(tac_template)
            if inv_template is not None and inv_template not in tac_templates:
                tac_templates.append(inv_template)
        #
        for tac_template in tac_templates:
            try:
                tactics = generate_tactics_from_template(tac_template, type_of_item)
                if len(tactics) > MAX_TACTIC_FROM_TEMPLATE:
                    tactics = random.sample(tactics, MAX_TACTIC_FROM_TEMPLATE)
            except:
                #print("Cannot generate tactics using", tac_template, ":", type_of_item)
                continue
            for tactic in tactics:
                tactic_set.add(tactic)
        #
        proof_finished = False
        tactic_trace = None
        for tactic in tactic_set:
            n_steps += 1
            if n_steps % 10000 == 0:
                print(f"{n_steps} steps executed")
            if n_steps > max_steps:
                break
            num_attempt = 0
            while num_attempt < MAX_NUM_DOJO_ATTEMPT:
                try:
                    test_state = dojo_run_tac(dojo, curr_state, tactic)
                    break
                except Exception as e:
                    num_attempt += 1
            if num_attempt == MAX_NUM_DOJO_ATTEMPT:
                continue
#             print("thm", theorem.full_name)
            if type(test_state) in [LeanError,TimeoutError,TacticResult,DojoCrashError,DojoHardTimeoutError,DojoInitError,ProofGivenUp]:
                continue
            elif type(test_state) == lean_dojo.interaction.dojo.ProofFinished:
                proof_finished = True
                if verbose and not theorem_proven:
                    print("ProofFinished")
                if test_state in state_dict:
                    _, parent_states, tactics, prefix = state_dict[test_state]
                    prefix += [tactic]
                    try:
                        tmp_state = dojo_run_tac(dojo, test_state, tactic)
                        a = tmp_state.pp
                    except Exception as e:
                        continue
                    state_dict[test_state] = (test_state, parent_states + [curr_state], tactics + [tactic], prefix)
#                     print("tacs", state_dict[test_state.pp][-2])
#                     print("prefix", state_dict[test_state.pp][-1])
                    prefix_2 = state_dict[curr_state.pp][3] + [tactic]
                    if len(prefix_2) < len(prefix):
                        state_dict[test_state][3] = prefix_2
                else:
                    complexity = 100000000
                    state_queue.push(test_state, complexity)
#                     print("tac4", [tactic])
                    state_dict[test_state] = (test_state, [curr_state], [tactic], state_dict[curr_state.pp][3] + [tactic])
                if exit_on_finish: break
            else:
                #print("TAC:", tactic, "STATE:", test_state.pp)
                if test_state.pp in state_dict:
                    _, parent_states, tactics, prefix = state_dict[test_state.pp]
                    prefix += [tactic]
                    try:
                        tmp_state = dojo_run_tac(dojo, test_state, tactic)
                        a = tmp_state.pp
                    except Exception as e:
                        continue
                    state_dict[test_state.pp] = (test_state, parent_states + [curr_state], tactics + [tactic], prefix)
#                     print("state2", test_state.pp)
#                     print("tacs2", state_dict[test_state.pp][-2])
#                     print("prefix2", state_dict[test_state.pp][-1])
#                     print("tac", tactic)
#                     print("\n")
#                     prefix_2 = state_dict[curr_state.pp][3] + [tactic]
                    prefix_2 = state_dict[curr_state.pp][3] + [tactic]
                    if len(prefix_2) < len(prefix):
                        state_dict[test_state.pp] = (test_state, parent_states + [curr_state], tactics + [tactic], prefix_2)
                else:
                    complexity = explore_state_complexity(test_state, base_complexity=base_complexity)
                    base_complexity += 1
                    state_queue.push(test_state, complexity  + randint(0, 4))
#                     print("state3", test_state.pp)
#                     print("tacs3", [tactic])
#                     print("prefix3", state_dict[curr_state.pp][3] + [tactic])
                    state_dict[test_state.pp] = (test_state, [curr_state], [tactic], state_dict[curr_state.pp][3] + [tactic])
        #
        if proof_finished:
            theorem_proven = True
            if exit_on_finish: break
        if n_steps > max_steps or (not theorem_proven and n_steps > max_steps / 3):
            break
        cur_time = time.time()
        if (cur_time - start_time) > max_time:
            print("Max run-time exceeded")
            break
    
    #     try:
    #         print("Dojo exiting ...")
    #         dojo.__exit__(None, None, None)
    #         print("Dojo exited")
    #         shutil.rmtree(dojo.tmp_dir)
    #         print(dojo.tmp_dir, "removed successfully.")
    #     except Exception as e:
    #         print(f"An error occurred when removing tmp dir: {e}")
    return state_dict, theorem_proven, tactic_trace

# Generate state pairs and their distances. 
# Probably useful if we hope to generate intermediate states (e.g., "have").
def yield_state_pairs(curr_state, leaf_state, state_dict, seen_states_pp, tactic_list, max_distance=8, max_parents=10):
    #     print("curr_state:", curr_state.pp)
    #     print("leaf_state:", leaf_state.pp)
    if type(curr_state) == lean_dojo.interaction.dojo.ProofFinished:
        _, parent_states, tactics, prefix = state_dict[curr_state]
    else:
        _, parent_states, tactics, prefix = state_dict[curr_state.pp]
    if len(seen_states_pp) > 1:
        yield (curr_state, leaf_state, len(seen_states_pp) - 1, tactic_list)
    if len(seen_states_pp) <= max_distance:
        num_parents_checked = 0
        for i in range(len(parent_states)):
            parent_state = parent_states[i]
            if parent_state.pp not in seen_states_pp:
                yield from yield_state_pairs(parent_state, 
                                             leaf_state, 
                                             state_dict, 
                                             seen_states_pp + [parent_state.pp],
                                             [tactics[i]] + tactic_list
                                            )
                num_parents_checked += 1
                if num_parents_checked >= max_parents:
                    break
    return

@timeout(3000)
def get_state_provability_data(dojo,
                               state_0,
                               theorem,
                               model_state,
                               tokenizer,
                               rag_index, 
                               all_tacs,
                               theorem_code=None, 
                               proof_tactics=None,
                               max_steps=100000, 
                               negative_ratio=1.0,
                               verbose=False):
    MAX_PROVEN_STATES = 1000
    start_time = datetime.now()
    print("Start exploring states at", start_time.time())
    state_dict, theorem_proven, tactic_trace = \
    explore_states(dojo,
                   state_0,
                   theorem,
                   model_state,
                   tokenizer,
                   rag_index, 
                   all_tacs,
                   theorem_code=theorem_code, 
                   proof_tactics=proof_tactics,
                   max_steps=max_steps,
                   exit_on_finish=False,
                   verbose=verbose)
    all_states = list(state_dict.values())
    print(f"{len(all_states)} states in total", datetime.now().time())
    proven_states = [x[0] for x in all_states if type(x[0]) == lean_dojo.interaction.dojo.ProofFinished]
    print(f"{len(proven_states)} proven states")
    state_pairs = []
    seen_state_pps = defaultdict(int)

    random.shuffle(proven_states)
    for i in range(len(proven_states)):
        proven_state = proven_states[i]
        #print(proven_state)
        pairs = list(yield_state_pairs(proven_state, proven_state, state_dict, [str(proven_state)], []))
        ancestor_state_dict = {}
        for ancestor, leaf, distance, tactic in pairs:
            if ancestor.pp not in ancestor_state_dict:
                ancestor_state_dict[ancestor] = (distance, tactic)
            else:
                if distance < ancestor_state_dist[ancestor]:
                    ancestor_state_dict[ancestor] = (distance, tactic)
        #print(ancestor_state_dist)
        #print(len(ancestor_state_dict), "ancestors found")
        for ancestor, (distance, tactic) in ancestor_state_dict.items():
            if seen_state_pps[ancestor.pp] < MAX_NUM_OUTPUT_PER_STATE:
                state_pairs.append((ancestor, proven_state, distance, tactic, state_dict[ancestor.pp][-1]))#last one is prefix
                seen_state_pps[ancestor.pp] += 1
            if (datetime.now() - start_time).seconds > 2900:
                return state_pairs, state_dict
        if i % 100 == 0 and i > 0:
            print(f"{i} proven states processed")
        if i > MAX_PROVEN_STATES:
            break
    #
    num_positive = len(state_pairs)
    # Add negative examples
    if num_positive > 0:
        num_negative = 0
        for tup in all_states:
            state = tup[0]
            if hasattr(state, 'pp') and state.pp not in seen_state_pps:
                state_pairs.append((state, random.choice(proven_states), -1, None, None))
                num_negative += 1
                if num_negative >= num_positive * negative_ratio:
                    break
                if (datetime.now() - start_time).seconds > 2900:
                    return state_pairs, state_dict
    return state_pairs, state_dict


def get_filename(filepath):
    # Extract the base name (e.g., 'example.txt' from '/path/to/example.txt')
    base_name = os.path.basename(filepath)
    # Split the base name and the extension and return just the base name
    file_name_without_extension = os.path.splitext(base_name)[0]
    return file_name_without_extension

def split_path(file_path):
    """
    Splits a given file path into its individual components (directories and file name).

    Args:
    file_path (str): The file path to split.

    Returns:
    list: A list of strings, each being a folder or file name in the path.
    """
    # Initialize an empty list to store path components
    parts = []

    # Keep splitting the path until there are no more directories left
    while True:
        parts.append(os.path.basename(file_path))
        file_path = os.path.dirname(file_path)

        # If the dirname is the same as the input, break the loop
        if file_path == os.path.dirname(file_path):
            if file_path:  # To avoid appending an empty string for root path
                parts.append(file_path)
            break

    # Since we've constructed the list from the file up to the root, reverse it
    parts.reverse()
    return parts

def get_all_theorems_processed(folder_path, verbose=True):
    # Enumerate all .pkl files in the folder
    output = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".pkl"):
            found = False
            file_path = os.path.join(folder_path, filename)
            fin = open(file_path, 'rb')
            while True:
                try:
                    result = pickle.load(fin)
                except Exception as e:
                    break
                file_path, full_name, theorem, state_pair = result
                if file_path not in output:
                    output[file_path] = set()
                output[file_path].add(full_name)
                found = True
            fin.close()
            if verbose and found:
                print(filename, len(output[file_path]), "theorems")
    return output

        
import posix_ipc
import time
import threading

class SystemSemaphore:
    def __init__(self, name, limit, max_hold_time=480):
        self.name = name
        self.limit = limit
        self.max_hold_time = max_hold_time

    def __enter__(self):
        self.lock = posix_ipc.Semaphore(f'/{self.name}', posix_ipc.O_CREAT, 0x384, self.limit)
        self.lock.acquire()
#         acquired = self.lock.acquire(self.acquire_timeout)
#         if not acquired:
#             raise TimeoutError(f"Failed to acquire semaphore within {self.acquire_timeout} seconds.")
        self.start_time = time.time()
        self.timer = threading.Timer(self.max_hold_time, self.release_semaphore)
        self.timer.start()

    def __exit__(self, _type, value, tb):
        self.timer.cancel()
        self.release_semaphore()

    def release_semaphore(self):
        self.lock.release()
        elapsed_time = time.time() - self.start_time
        print(f"Warning: Semaphore was held for {elapsed_time:.2f} seconds, exceeding the maximum hold time of {self.max_hold_time} seconds.")
        #raise FooTimeoutError(os.strerror(errno.ETIME))
        

# @timeout(300)
# def dojo_enter(theorem):
#     return Dojo(theorem).__enter__()


@ray.remote(num_cpus=1,num_gpus=0.03)
def compute_provability_training_data_remote(file_path, output_path, previous_output_path=None):
    try:
        worker_id = random.randint(0, 100)
        print("Worker", worker_id, "on", file_path)
        cuda_version = subprocess.run(['nvcc', '--version'], capture_output=True, text=True).stdout
        print(cuda_version)
        print("CUDA:", torch.cuda.is_available())
        print("os.environ['LD_LIBRARY_PATH']", os.environ['LD_LIBRARY_PATH'])
        print("Worker", worker_id, "Loading repo...")
        repo = LeanGitRepo(
            #"https://github.com/xiaoxin-yin/math-in-lean",
            #"20077bcd4392317ddb9605404fda3a85e40e8956"
            "https://github.com/leanprover-community/mathlib4",
            "27c6744e1c0e25d676be5eb252cd4b6d30c6acc7",
        )
        #fin = open('/home/mcwave/code/automath/atp/datasets/train_traced_theorems_repo_math_in_lean.pkl', 'rb')
        #train_traced_theorems = pickle.load(fin)
        #fin.close()
        fin = open('/home/mcwave/code/automath/atp/datasets/remaining_theorems_repo_mathlib4_20240716.pkl', 'rb')
        train_theorems = pickle.load(fin)
        fin.close()
        #
        #     if previous_output_path is not None:
        #         print("Getting previously computed theorems")
        #         processed_theorems = get_all_theorems_processed(previous_output_path)
        #         remaining_traced_theorems = dict()
        #         for train_file_path, traced_theorems in train_traced_theorems.items():
        #             if train_file_path not in processed_theorems:
        #                 remaining_traced_theorems[train_file_path] = traced_theorems
        #                 continue
        #             remaining_theorems = dict()
        #             for full_name, thm in traced_theorems.items():
        #                 if full_name not in processed_theorems[train_file_path]:
        #                     remaining_theorems[full_name] = thm
        #                 else:
        #                     remaining_theorems = dict()
        #             print(train_file_path, len(traced_theorems), "->", len(remaining_theorems))
        #             remaining_traced_theorems[train_file_path] = remaining_theorems
        #         train_traced_theorems = remaining_traced_theorems
        #
        print("Worker", worker_id, "Loading models...")
        # Load pre-trained BERT tokenizer and model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model_state = torch.load('/home/mcwave/code/automath/atp/datasets/rag_tactic_templates/bert_embeder_state-batch64-60k-loss0047.model')
        #model_tac   = torch.load('/home/mcwave/code/automath/atp/datasets/rag_tactic_templates/bert_embeder_tac-batch64-60k-loss0047.model')
        device = model_state.device
        print(device)
        #
        print("Worker", worker_id, "Loading faiss index ...")
        index = faiss.read_index('/home/mcwave/code/automath/atp/datasets/rag_tactic_templates/faiss_index_bert_embeds-batch64-60k-loss0047.idx')
        fin = open('/home/mcwave/code/automath/atp/datasets/rag_tactic_templates/tac_template_freq.json', 'r')
        tac_template_freq = json.load(fin)
        fin.close()
        tacs = list(tac_template_freq.keys())
        #
        print("Worker", worker_id, "Computing provability ...")
        file_name = get_filename(file_path)
        parts = split_path(file_path)
        output_file_name = '__'.join(parts)
        output_file_path = output_path + output_file_name + '.pkl'
        print("Writing to", output_file_path)
        fout = open(output_file_path, 'wb')
        theorems = train_theorems[file_path]
        #
        results = []
        num_theorems_processed = 0
        dojo = None
        for full_name, val in theorems.items():
            print("val_test", val)
            thm, theorem_code, tactics = val
            print("TACTICS:", tactics)
            #print('Start Loading Theorem:', full_name, theorem_code)
            theorem = Theorem(repo, file_path, thm.full_name)
            # Getting Dojo and state_0
            # For some theorems, it might take a few minutes.
            print("Worker", worker_id, "trying to enter theorem", theorem)
            entered = False
            if dojo is None:
                try:
                    print("Worker", worker_id, "trying to get into critical section", datetime.now().time())
                    with SystemSemaphore('dojolock37', 1):
                        print("Worker", worker_id, f'Process {os.getpid()} has exclusive access to the critical section!')
                        try:
                            print("Start entering", theorem)
                            dojo, state_0 = Dojo(theorem, copy_tree_on_enter=False).__enter__() #dojo_enter(theorem)
                            entered = True
                            print("Finish entering", theorem)
                        except lean_dojo.interaction.dojo.DojoInitError:
                            print("DojoInitError")
                except Exception as e:
                    print("Exception when trying to get into critical section:", e)
            else:
                try:
                    print(f"Continue entering {theorem} at {datetime.now().time()}")
                    dojo.entry = theorem
                    _, state_0 = dojo.__enter__()
                    entered = True
                    print(f"Finish entering {theorem}")
                except Exception as e:
                    print("Exception when entering theorem:", e)
            if not entered:
                print(f"Failed to enter {theorem}")
                result = (file_path, full_name, theorem, None)
                pickle.dump(result, fout)
                fout.flush()
                continue
            #
            # The main computation
            state_pairs, _ = \
                get_state_provability_data(dojo,
                                           state_0,
                                           theorem,
                                           model_state,
                                           tokenizer,
                                           index,
                                           tacs,
                                           theorem_code=theorem_code,
                                           proof_tactics=tactics,
                                           max_steps=60000,
                                           verbose=True)
            print(f"{len(state_pairs)} state pairs found in {full_name}, {file_path}")
            if len(state_pairs) == 0:
                result = (file_path, full_name, theorem, None)
                pickle.dump(result, fout)
            #
            i = 0
            for state_pair in state_pairs:
                result = (file_path, full_name, theorem, state_pair)
                results.append(result)
                pickle.dump(result, fout)
                i += 1
            fout.flush()
            #
            num_theorems_processed += 1
        #
        # Processed all theorems in this file_path
        fout.close()
        logger.info(f"Finished processing {file_path}, Returning {len(results)}")
        try:
            logger.info("Dojo exiting ...")
            dojo.__exit__(None, None, None)
            logger.info("Dojo exited")
            shutil.rmtree(dojo.tmp_dir)
            logger.info(f"{dojo.tmp_dir} removed successfully.")
        except Exception as e:
            logger.info(f"An error occurred when removing tmp dir: {e}")
        #
        return len(results)
    except Exception as e:
        logger.info(str(e))
        return 0

#Rename datasets/provability/rag and create new rag folder
#Regenerate remaining_theorems file
#Start process
    
#nohup python utils/lean_rag_utils.py > datasets/test_log.txt 2>&1 &
#tail -f datasets/test_log.txt

    

def main() -> int:
#     random.seed(42)
    
    fin = open('/home/mcwave/code/automath/atp/datasets/remaining_theorems_repo_mathlib4_20240716.pkl', 'rb')
    train_theorems = pickle.load(fin)
    fin.close()
    
    output_folder = '/home/mcwave/code/automath/atp/datasets/provability/rag/'
    #compute_provability_training_data_remote('.lake/packages/mathlib/Mathlib/Topology/Algebra/Module/Basic.lean', output_folder)
    
    all_file_paths = [] 
    for file_path, theorems in train_theorems.items():
        if theorems is not None and len(theorems) > 0:
            all_file_paths.append(file_path)
    random.shuffle(all_file_paths)
    
    #
    # Set the environment variable to disable log deduplication
    os.environ['RAY_record_all_task_output'] = '1'
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Ensure Ray is not already initialized
    if ray.is_initialized():
        ray.shutdown()
    #
    # Start Ray with custom configuration
    #logging.basicConfig(level=logging.ERROR, filename='datasets/ray_logs.log')
    ray.init(
        num_gpus=1,
        num_cpus=20,
        _memory=(192 * 1024 * 1024 * 1024),  # For example, limit Ray to 4 GB of RAM
        object_store_memory=(8 * 1024 * 1024 * 1024),  # Set object store memory to 2 GB
        #logging_level=logging.ERROR,
        #log_to_driver=True
    )
    output_folder = '/home/mcwave/code/automath/atp/datasets/provability/rag/'

    # Submit tasks
    result_ids = [compute_provability_training_data_remote.remote(file_path, output_folder) for file_path in all_file_paths]

    # Fetch results
    results = ray.get(result_ids)

    # Optionally, shut down Ray if you're done with all computations
    ray.shutdown()

    # Display results
    print(results)

# nohup python utils/lean_rag_utils.py > datasets/test_log.txt 2>&1 &
# 
    
if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
    