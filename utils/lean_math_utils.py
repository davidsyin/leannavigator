import json
import os
import sys
import re
import itertools
import heapq
import errno
import functools
import signal
import time

import lean_dojo


lean_tactics = [
    "all_goals", "any_goals", "apply", "assumption", "assumption'", "cases", 
    "change", "clear", "contradiction", "constructor", "dec_trivial", "exact",
    "existsi", "ext", "fapply", "have", "induction", "injection", "intro", "intros",
    "left", "let", "library_search", "match_target", "refine", "repeat", "replace",
    "revert", "rewrite", "rw", "rintro", "rintros", "rcases", "simp", "solve_by_elim",
    "split", "subst", "tactic.trace", "trivial", "use", "with_cases", "rfl", "refl", 
    "simp_all", "specialize", "apply_instance", "norm_num", "norm_cast", "ring", 
    "ring2", "linarith", "omega", "tauto", "by_contradiction", "by_cases", 
    "trace_state", "work_on_goal", "swap", "rotate", "rename", "guard_expr_eq", 
    "set_goals", "clear_except", "apply_with", "run_tac", "done", "unfold", "unfold1",
    "fail_if_success", "success_if_fail", "infer_type", "expr", "retrieve", "push_neg",
    "contrapose", "iterate", "repeat", "try", "skip", "solve1", "abstract", "generalize",
    "guard_hyp", "guard_target", "guard_hyp_nums", "guard_tags", "guard_proof_term", 
    "guard_expr_strict", "discharge", 'obtain', 'simpa', 'rwa', 'simp_rw', 'haveI',
    'by_contra','lift','letI','dsimp','split_ifs','ext1','convert','tfae_have','funext',
    'congr','filter_upwards','choose','field_simp','aesop','nth_rw','conv_lhs','simp_arith',
    'erw','delta','gcongr','positivity','right','infer_instance','abel','nontriviality','push_cast',
    'borelize','inhabit','ring_nf','nlinarith','fin_cases','trans','measurability','exact_mod_cast',
    'rename_i','calc','decide','symm','exfalso','aesop_cat','subst_vars','ac_rfl','continuity',
    'tfae_finish','rotate_left','classical','fconstructor','clear_value','conv_rhs','next',
    'assumption_mod_cast','substs','exists','mono','interval_cases','show','bitwise_assoc_tac',
    'triv','mfld_set_tac','beta_reduce','abel_nf','case','simp_wf','set','wlog','conv'
]


lean_keywords = [
    "abbrev", "axiom", "begin", "builtin", "by", "calc", "check", "coercion", "constant", "constructor", 
    "def", "definition", "derive", "do", "else", "end", "example", "export", "extends", "extern", 
    "if", "import", "inductive", "infix", "infixl", "infixr", "instance", "let", "macro", "match", 
    "meta", "mutual", "namespace", "noncomputable", "notation", "opaque", "open", "partial", "postfix", 
    "prefix", "prelude", "private", "protected", "public", "quotation", "reserve", "scoped", "section", 
    "set_option", "structure", "suffices", "tactic", "theorem", "universe", "universes", "using", "variable", 
    "variables", "where", "with", "without"
]


VARIABLE_PATTERN = r'[a-zA-Z_\'\u03b1-\u03c9][a-zA-Z_\'\u03b1-\u03c9\d₀-₉]*'


class FooTimeoutError(Exception):
    pass

def timeout(seconds=1, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise FooTimeoutError(error_message)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator


def parse_lean_definition(lean_code):
    # Regular expression to capture the name, signature, and content
    pattern = r"\bdef\s+(\w+)\s*(.*?)\s*:=\s*(.*)"
    match = re.search(pattern, lean_code, re.DOTALL)
    
    if match:
        name = match.group(1).strip()
        signature = match.group(2).strip()
        content = match.group(3).strip()
        if signature.endswith('Prop'):
            signature = signature[:-4].strip()
        if signature.endswith(':'):
            signature = signature[:-1].strip()
        return name, signature, content
    else:
        # Regular expression to capture the name, signature, and content
        pattern = r"\bdef\s+(\w+)\s*(.*)"
        match = re.search(pattern, lean_code, re.DOTALL)
        if match:
            name = match.group(1).strip()
            return name, '', ''
        return None, None, None

def classify_lean_element(line):
    VARIABLE_TYPES = ['ℝ']
    
    # Clean up the line
    line = line.strip()
    if not line:
        return None, None, None

    # Split the line around the ':'
    parts = line.split(':', 1)
    if len(parts) != 2:
        return None, None, None  # Skip malformed lines

    # Clean up names and types
    name_part, type_info = parts[0].strip(), parts[1].strip()

    if type_info.startswith('Type'):
        return 'type', name_part.split(), type_info
    
    if type_info.startswith('Set'):
        return 'set', name_part.split(), type_info
    
    if type_info.startswith(('∀', '∃', '¬')):
        return 'hypothesis', name_part.split(), type_info
    
    if type_info in VARIABLE_TYPES:
        return 'variable', name_part.split(), type_info
    
    # Check for function definitions or lambda functions
    if '→' in type_info or 'fun' in type_info or ':=' in type_info:
        # Check if it's a hypothesis disguised as a function (look for logical operators)
        if '∀' in type_info or '∃' in type_info:
            return 'hypothesis', name_part.split(), type_info
        return 'function', name_part.split(), type_info

    # Check for propositions or logical statements involving quantifiers
    if 'Prop' in type_info or '¬' in type_info or '∈' in type_info or '∉' in type_info:
        return 'hypothesis', name_part.split(), type_info

    # Handle multiple variables on the same line
    names = name_part.split()
    return 'unknown', names, type_info

def classify_lean_elements(lean_state):
    category_of_name = {}
    type_info_of_name = {}
    lines = lean_state.strip().split('\n')
    for line in lines:
        category, names, type_info = classify_lean_element(line)
        if category is None or names is None:
            continue
        for name in names:
            category_of_name[name] = category
            type_info_of_name[name] = type_info
    
    names = list(category_of_name.keys())
    for name in names:
        if category_of_name[name] == 'unknown' and\
            type_info_of_name[name] in names and\
            category_of_name[type_info_of_name[name]] == 'type':
            category_of_name[name] = 'variable'

    return category_of_name, type_info_of_name


def is_variable_name(token):
    variable_pattern = VARIABLE_PATTERN
    match = re.match(variable_pattern, token)
    return match


def tokenize_lean_tactic(tactic):
    # Define the regex patterns
    variable_pattern = VARIABLE_PATTERN
    operator_pattern = r'[:=<>∈∉≤≥+\-*/^∀∃¬⁻¹]+'
    punctuation_pattern = r'[,⟨⟩\[\](){}|]'
    comment_pattern = r'--.*'

    # Remove comments from the tactic
    tactic = re.sub(comment_pattern, '', tactic)

    # Tokenize the tactic
    tokens = []
    while tactic:
        # Check for variables
        match = re.match(variable_pattern, tactic)
        if match:
            tokens.append(match.group())
            tactic = tactic[match.end():]
            continue

        # Check for operators
        match = re.match(operator_pattern, tactic)
        if match:
            tokens.append(match.group())
            tactic = tactic[match.end():]
            continue

        # Check for punctuation
        match = re.match(punctuation_pattern, tactic)
        if match:
            tokens.append(match.group())
            tactic = tactic[match.end():]
            continue

        # Add remaining characters as separate tokens
        tokens.append(tactic[0])
        tactic = tactic[1:]

    return tokens


def transform_bracket_contents(contents, definition_dict, bracket='[]'):
    # Split the contents within the brackets by commas and strip spaces
    items = map(str.strip, contents.split(","))
    transformed = []
    for item in items:
        if item in definition_dict:
            transformed.append("{" + definition_dict[item] + "}")
        else:
            transformed.append(item)
    return bracket[0] + ", ".join(transformed) + bracket[-1]

def parse_lean_state_and_tactic(state_before, tactic, custom_defs=None):

    # Step 1: Parse the state_before to create a dictionary of definitions
    definition_dict, _ = classify_lean_elements(state_before)

    # Step 2: Parse the tactic
    tactic_parts = tokenize_lean_tactic(tactic)
    if len(tactic_parts) == 1:  # tactic without arguments
        return tactic  # no transformation needed

    # Step 3: Check and transform arguments
    command = None
    transformed_parts = []
    new_var_idx = 0
    while 'nvar' + str(new_var_idx) in tactic_parts:
        new_var_idx += 1
    for tp in tactic_parts:
        if tp in definition_dict:
            transformed_parts.append('{' + definition_dict[tp] + '}')
        elif custom_defs is not None and tp in custom_defs:
            transformed_parts.append('{custom_def:' + custom_defs[tp] + '}')
        else:
            if command is None and tp in lean_tactics:
                command = tp
            elif tp != ' ' \
                and command in ['intro', 'rintro', 'ext', 'ext1', 'cases', 'rcases', 'by_contra', 'by_cases', 'funext', 'congr','gcongr'] \
                and is_variable_name(tp):
                if tp not in lean_tactics and tp not in lean_keywords:
                    transformed_parts.append('{nvar' + str(new_var_idx) + '}')
                    new_var_idx += 1
                    continue
            transformed_parts.append(tp)
    
    return ''.join(transformed_parts)

@timeout(1)
def generate_tactics_from_template(template, type_of_item):
    items_of_type = {'unknown': [], 'variable': [], 'hypothesis': []}
    for item, type_ in type_of_item.items():
        if type_ in items_of_type:
            items_of_type[type_].append(item)
            
    # Replace {nvar*} with nvar*
    existing_new_var_indices = [x for x in range(32) if 'nvar'+str(x) in type_of_item]
    if len(existing_new_var_indices) == 0:
        additional_new_var_idx = 0
    else:
        additional_new_var_idx = max(existing_new_var_indices) + 1
    for new_var_idx in range(32):
        tmp = '{nvar' + str(new_var_idx) + '}'
        if tmp in template:
            template = template.replace(tmp, 'nvar' + str(additional_new_var_idx))
            additional_new_var_idx += 1
        
    # Identify the placeholders and their types
    parts = template.split('{')
    fixed_parts = [parts[0]]
    types = []
    
    for part in parts[1:]:
        typ, rest = part.split('}')
        types.append(typ)
        fixed_parts.append(rest)
    
    # Generate all possible combinations of replacements
    replacement_lists = [items_of_type[typ] for typ in types]
    #print(replacement_lists)
    combinations = itertools.product(*replacement_lists)
    
    # Construct the sentences with all possible combinations
    sentences = []
    for combination in combinations:
        sentence = fixed_parts[0]
        for item, fixed_part in zip(combination, fixed_parts[1:]):
            sentence += item + fixed_part
        sentences.append(sentence)
    
    return sentences

class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0  # this index helps to compare items with the same priority

    def size(self):
        return len(self._queue)

    def push(self, item, priority):
        # The priority queue is based on a min-heap, so we use negative priority to ensure that
        # the highest priority has the lowest number.
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1

    def pop(self):
        # Returns items with the highest priority
        if len(self._queue) == 0:
            return None
        return heapq.heappop(self._queue)[-1]  # Returns only the item, discarding its priority and index


def state_complexity(state):
    complexity = 0
    lines = state.pp.split('\n')
    n_targets = 0
    for line in lines:
        if line.startswith("⊢"):
            complexity = max(complexity, len(line[1:].strip()))
            n_targets += 1
    return complexity + n_targets - 1

# Save traced_theorems to a pickle file since they are only data we need
def generate_train_file_theorems(train_tac_templates_path, output_path):
    fin = open(train_tac_templates_path, 'r')
    train_tac_templates = json.load(fin)
    fin.close()
    #
    train_file_theorems = {}
    for i in range(len(train_tac_templates)):
        full_name = train_tac_templates[i][2]
        file_path = train_tac_templates[i][3]
        if file_path not in train_file_theorems:
            train_file_theorems[file_path] = set()
        train_file_theorems[file_path].add(full_name)
    #
    train_traced_theorems = dict()
    for file_path, full_names in train_file_theorems.items():
        traced_file = traced_repo.get_traced_file(file_path)
        premises = traced_file.get_premise_definitions()
        results = []
        for premise in premises:
            full_name = premise['full_name']
            if full_name in full_names:
                thm = traced_file.get_traced_theorem(full_name)
                thm.comments.insert(0, premise['code'])
                if file_path not in train_traced_theorems:
                    train_traced_theorems[file_path] = dict()
                train_traced_theorems[file_path][full_name] = thm
    #
    for file_path, full_names in train_file_theorems.items():
        if file_path not in train_traced_theorems:
            print(file_path, "not found!")
            continue
        for full_name in full_names:
            if full_name not in train_traced_theorems[file_path]:
                print(full_name, "not found in", file_path)
    #
    fout = open(output_path, 'wb')
    pickle.dump(train_traced_theorems, fout)
    fout.close()

#train_tac_templates_path = '/home/mcwave/code/automath/atp/datasets/tac_templates_in_files/train_tac_templates.json'
#output_path = '/home/mcwave/code/automath/atp/datasets/train_traced_theorems_repo_math_in_lean.pkl'
#generate_train_file_theorems(train_tac_templates_path, output_path)



def main() -> int:
    # Example usage
    lean_state = """
    case h₁
    f g : ℝ → ℝ
    ⊢ f x ≤ a

    α : Type u_1
    P : α → Prop
    Q : Prop
    h : ¬∀ (x : α), P x
    h' : ¬∃ x, ¬P x
    x : α
    h'' : ¬P x
    a b : ℝ
    f✝ : ℝ → ℝ
    h1 : ∀ {f : ℝ → ℝ}, Monotone f → ∀ {a b : ℝ}, f a ≤ f b → a ≤ b
    f : ℝ → ℝ := fun x => 0
    α1 : Type u_1
    s t : Set α
    xstu : x ∈ (s \ t) \ u
    xs : x ∈ s
    xnt : x ∉ t
    xnu : x ∉ u
    xu : x ∈ u
    ubf : FnHasUb f
    ι : Type u_1
    R : Type u_2
    inst✝ : CommRing R
    I : Ideal R
    J : ι → Ideal R
    i : ι
    hs : (∀ j ∈ s, I + J j = 1) → I + ⨅ j ∈ s, J j = 1
    hf : ∀ j ∈ insert i s, I + J j = 1
    inst✝ : CommRing R
    """

    category_of_name, _ = classify_lean_elements(lean_state)
    for name, category in category_of_name.items():
        print(name, category)
        
    test_cases = [
        "intro h''",
        "rintro y ⟨x, ⟨xs, xt⟩, rfl⟩",
        "x = exp (log x) := by rw [exp_log xpos]",
        "have ε2pos : 0 < ε / 2 := by linarith",
        "rw [← h1 fx₂eq]",
        "rintro y ⟨⟨x₁, x₁s, rfl⟩, ⟨x₂, x₂t, fx₂eq⟩⟩",
        "use y*h1*y⁻¹",
        "have : j ∉ f j := by rwa [h2] at h'",
        "rw [mem_map] at * -- Lean does not need this line",
        "have h₀ : 0 < 1 + exp a := by linarith [exp_pos a]",
        "_ ≤ |s n - a| + |a| := (abs_add _)_",
        "|a - b| = |(-(s N - a)), + (s N - b)| := by",
        "∃ a, FnLb f a", 
        "¬¬∀ x, a ≤ f x",
        "s ∩ t ∪ s ∩ u ⊆ s ∩ (t ∪ u)"
    ]

    expected_outputs = [
        ['intro', "h''"],
        ['rintro', 'y', '⟨', 'x', ',', '⟨', 'xs', ',', 'xt', '⟩', ',', 'rfl', '⟩'],
        ['x', '=', 'exp', '(', 'log', 'x', ')', ':=', 'by', 'rw', '[', 'exp_log', 'xpos', ']'],
        ['have', 'ε2pos', ':', '0', '<', 'ε', '/', '2', ':=', 'by', 'linarith'],
        ['rw', '[', '←', 'h1', 'fx₂eq', ']'],
        ['rintro', 'y', '⟨', '⟨', 'x₁', ',', 'x₁s', ',', 'rfl', '⟩', ',', '⟨', 'x₂', ',', 'x₂t', ',', 'fx₂eq', '⟩', '⟩'],
        ['use', 'y', '*', 'h1', '*', 'y', '⁻', '¹'],
        ['have', ':', 'j', '∉', 'f', 'j', ':=', 'by', 'rwa', '[', 'h2', ']', 'at', "h'"],
        ['rw', '[', 'mem_map', ']', 'at', '*'],
        ['have', 'h₀', ':', '0', '<', '1', '+', 'exp', 'a', ':=', 'by', 'linarith', '[', 'exp_pos', 'a', ']'],
        ['_', '≤', '|', 's', 'n', '-', 'a', '|', '+', '|', 'a', '|', ':=', '(', 'abs_add', '_', ')', '_'],
        ['|', 'a', '-', 'b', '|', '=', '|', '(', '-', '(', 's', 'N', '-', 'a', ')', ')', ',', '+', '(', 's', 'N', '-', 'b', ')', '|', ':=', 'by'],
        ['∃', 'a', ',', 'FnLb', 'f', 'a'],
        ['¬', '¬', '∀', 'x', ',', 'a', '≤', 'f', 'x'],
        ['s', '∩', 't', '∪', 's', '∩', 'u', '⊆', 's', '∩', '(', 't', '∪', 'u', ')']
    ]

    num_mismatch = 0
    for i in range(len(test_cases)):
        output = tokenize_lean_tactic(test_cases[i])
        output = [x for x in output if x != ' ']
        if output != expected_outputs[i]:
            print("Case",i,"mismatches:",test_cases[i])
            print("Expected:", expected_outputs[i])
            print("Actual:  ", output)
            num_mismatch += 1

    print(num_mismatch, "mismatches")
    
    
    # Example usage for both scenarios
    state_before = """
    case h₁
    f g : ℝ → ℝ
    a b : ℝ
    hfa : FnUb f a
    hgb : FnUb g b
    x : ℝ
    ⊢ f x ≤ a

    case h₂
    f g : ℝ → ℝ
    a b : ℝ
    hfa : FnUb f a
    hgb : FnUb g b
    x : ℝ
    ⊢ g x ≤ b
    """
    tactic = "apply hfa x"
    print(parse_lean_state_and_tactic(state_before, tactic))  # should be "apply {FnUb f a}"

    state_before = """
    f g : ℝ → ℝ
    ef : FnEven f
    ε2pos : FnEven g
    x : ℝ
    ⊢ f x + g x = f (-x) + g (-x)
    """
    tactic = "rw [f, eg]"
    print(parse_lean_state_and_tactic(state_before, tactic))  # should be "rw [{FnEven f}, {FnEven g}]"


    state_before = """
    f g : ℝ → ℝ
    ⊢ f x + g x = f (-x) + g (-x)
    """
    tactic = "rintro y ⟨x, ⟨xs,xt⟩, rfl⟩"
    print(parse_lean_state_and_tactic(state_before, tactic))  # should be "rw [{FnEven f}, {FnEven g}]"
    
    type_of_item = {
        'a_pos': 'unknown',
        'b_neg': 'unknown',
        'x': 'variable',
        'y': 'variable',
        'h': 'hypothesis',
        'h1': 'hypothesis',
        'nvar1': 'hypothesis'
    }

    templates = ["apply le_trans", "apply {unknown}", "intro {variable} {nvar0}", "intro {unknown} with [{variable}, {variable}] in {hypothesis}"]
    for template in templates:
        print("Template:", template)
        tactics = generate_tactics_from_template(template, type_of_item)
        if template == 'apply le_trans':
            assert len(tactics) == 1
            assert tactics[0] == 'apply le_trans'
        elif template == 'apply {unknown}':
            assert len(tactics) == 2
            assert tactics[0] == 'apply a_pos'
            assert tactics[1] == 'apply b_neg'
        elif template == 'intro {variable} {nvar0}':
            assert len(tactics) == 2
            assert tactics[0] == 'intro x nvar2'
            assert tactics[1] == 'intro y nvar2'
        elif template == "intro {unknown} with [{variable}, {variable}] in {hypothesis}":
            assert len(tactics) == 24
    
    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit

    

