from dsl import *
from arc_types import *
from constants import *
import solvers
import solvers2
# import solvers3
from main import get_functions, get_data, list_to_tuple
import re
import inspect
import json
import numpy as np
from tqdm import tqdm

import multiprocessing
from joblib import Parallel, delayed

def replace_variable(v, offset, i_replacement=None, o_replacement=None, first_solver=False):
    pattern = r'^x([1-9][0-9]{0,2})$'
    match = re.match(pattern, v)
    if not match or first_solver:
        if v == 'I' and i_replacement:
            return i_replacement
        if v == 'O' and o_replacement:
            return o_replacement
        return v
    new_index = int(match.group(1)) + offset
    return f'x{new_index}'

def update_line(line, offset, replace_O=False, i_replacement=None, first_solver=False):
    o_replacement = f'x{offset+1}' if replace_O else None
    variable, call = line.lstrip().split(' = ')
    function, args = call.split('(')
    args = [arg.strip() for arg in args.split(')')[0].split(',')]
    new_args = [replace_variable(arg, offset, i_replacement, o_replacement, first_solver) for arg in args]
    new_variable = replace_variable(variable.strip(), offset, i_replacement, o_replacement, first_solver)
    new_function = replace_variable(function.strip(), offset, i_replacement, o_replacement, first_solver)
    new_line = f'    {new_variable} = {new_function}({", ".join(new_args)})'
    return new_line

def concat_solvers(s1, s2):
    try:
        lines1 = s1.split('\n')
        lines2 = s2.split('\n')
        s1key = lines1[0].split('solve_')[1].split('(')[0]
        s2key = lines2[0].split('solve_')[1].split('(')[0]
        s1finalx= lines1[-4].split(' = ')[0].lstrip()
        s1finalcount = int(s1finalx.split('x')[1]) if s1finalx[0] == 'x' else 0

        # replace first solver's output with a new variable
        lines1[-3] = update_line(lines1[-3], s1finalcount, replace_O=True, first_solver=True)
        offset = s1finalcount+1
        i_replacement = f'x{offset}'


        news2 = []
        for line in lines2[1:-2]:
            new_line = update_line(line, offset, i_replacement=i_replacement)
            news2.append(new_line)
        news2.append(lines2[-2])
        news2.append(lines2[-1])

        new_def = f'def solve_{s1key}_{s2key}(I):'
        for l1 in lines1[1:-2]:
            new_def += '\n' + l1
        for l2 in news2:
            new_def += '\n' + l2

        return new_def, f'solve_{s1key}_{s2key}'
    except:
        return None

def generate_new_output(solver, inputs, solvername):
    new_outputs = []
    try:
        exec(solver, globals())
        for i in inputs:
            input = list_to_tuple(i)
            new_output = globals()[solvername](input)
            new_outputs.append(np.array(new_output).tolist())
    except Exception as e:
        # print(e)
        return None
    return new_outputs

def concat_and_generate(data):
    (key1, definition1, key2, definition2, train_input, test_input) = data
    k1 = key1.split('solve_')[1]
    k2 = key2.split('solve_')[1]

    k_list = k1.split('_')
    key_match = [k == k2 for k in k_list]
    if any(key_match):
        return (None, None, None)
    k_list.append(k2)
    
    new_def, new_key = concat_solvers(definition1, definition2)

    if not new_def:
        # print('no new def')
        return (None, None, None)


    new_train_outputs = generate_new_output(new_def, train_input, new_key)
    if not new_train_outputs:
        # print('no new train outputs')
        return (None, None, None)
    
    new_test_outputs = generate_new_output(new_def, test_input, new_key)
    if not new_test_outputs:
        # print('no new test outputs')
        return (None, None, None)

    final_key = '_'.join(k_list)
    new_data = {
        'train': [{'input': i, 'output': o} for i, o in zip(train_input, new_train_outputs)],
        'test': [{'input': i, 'output': o} for i, o in zip(test_input, new_test_outputs)]
    }

    return (new_def, final_key, new_data)

def combine(definitions1, definitions2, solver_filename, data_filename):
    get_key = lambda k: k.split('solve_')[1]
    get_first_key = lambda ks: ks.split('_')[0]

    data = get_data(train=True)

    n = len(definitions1)*len(definitions2)
    num_cores = multiprocessing.cpu_count()
    inputs = tqdm([(key1, 
                    definition1, 
                    key2, 
                    definition2, 
                    [ex['input'] for ex in data[get_first_key(get_key(key1))]['train']], 
                    [ex['input'] for ex in data[get_first_key(get_key(key1))]['test']]) 
                    for key1, definition1 in definitions1.items() for key2, definition2 in definitions2.items()],
                    total=n)
    print(n)
    print(len(inputs))
    
    # outputs = Parallel(n_jobs=num_cores, verbose=0)(delayed(concat_and_generate)(inp) for inp in inputs)
    outputs = [concat_and_generate(inp) for inp in inputs]

    full_data = {}

    with open(solver_filename, 'w') as solve_file:
        for output in outputs:
            if output[0] is None:
                continue
            (new_definition, new_key, new_data) = output

            solve_file.write(new_definition)
            solve_file.write('\n')

            full_data[new_key] = new_data

    print(len(full_data))
    with open(data_filename, 'w') as data_file:
        data_file.write(json.dumps(full_data))



def combine_3():
    definitions = {
        function: inspect.getsource(getattr(solvers, function)) \
        for function in get_functions(solvers.__file__)
    }
    
    definitions2 = {
        function: inspect.getsource(getattr(solvers2, function)) \
        for function in get_functions(solvers2.__file__)
    }

    solver_filename = 'solvers3.py'
    data_filename = 'data3.json'

    combine(definitions2, definitions, solver_filename, data_filename)



    # with open('solvers3.py', 'w') as solve_file:
    #     with open('data3.json', 'w') as data_file:
    #         try:
    #             data = get_data(train=True)
    #             new_data = {}
    #             for key1, definition1 in tqdm(definitions2.items()):
    #                 for key2, definition2 in definitions.items():
    #                     k1 = key1.split('solve_')[1]
    #                     k2 = key2.split('solve_')[1]

    #                     k1_list = k1.split('_')
    #                     k1_1 = k1_list[0]
    #                     k1_2 = k1_list[1]

    #                     if k1_1 == k2 or k1_2 == k2:
    #                         continue
                        
    #                     new_def, new_key = concat_solvers(definition1, definition2)

    #                     if not new_def:
    #                         # print('no new def')
    #                         continue
    #                     task_train_inputs = [ex['input'] for ex in data[k1_1]['train']]
    #                     new_train_outputs = generate_new_output(new_def, task_train_inputs, new_key)
    #                     if not new_train_outputs:
    #                         # print('no new train outputs')
    #                         continue
    #                     task_test_inputs = [ex['input'] for ex in data[k1_1]['test']]
    #                     new_test_outputs = generate_new_output(new_def, task_test_inputs, new_key)
    #                     if not new_test_outputs:
    #                         # print('no new test outputs')
    #                         continue

    #                     new_data[f'{k1_1}_{k1_2}_{k2}'] = {
    #                         'train': [{'input': i, 'output': o} for i, o in zip(task_train_inputs, new_train_outputs)],
    #                         'test': [{'input': i, 'output': o} for i, o in zip(task_test_inputs, new_test_outputs)]
    #                     }

    #                     solve_file.write(new_def)
    #                     solve_file.write('\n')

    #             print(len(new_data))
    #             data_file.write(json.dumps(new_data))
    #         except Exception as e:
    #             # print(e)
    #             pass


def combine_2():
    definitions = {
        function: inspect.getsource(getattr(solvers, function)) \
            for function in get_functions(solvers.__file__)
    }


    with open('solvers2.py', 'w') as solve_file:
        with open('data2.json', 'w') as data_file:
            try:
                data = get_data(train=True)
                new_data = {}
                for key1, definition1 in definitions.items():
                    for key2, definition2 in definitions.items():
                        if key1 == key2:
                            continue
                        k1 = key1.split('solve_')[1]
                        k2 = key2.split('solve_')[1]
                        new_def, new_key = concat_solvers(definition1, definition2)
                        if not new_def:
                            # print('no new def')
                            continue
                        task_train_inputs = [ex['input'] for ex in data[k1]['train']]
                        new_train_outputs = generate_new_output(new_def, task_train_inputs, new_key)
                        if not new_train_outputs:
                            # print('no new train outputs')
                            continue
                        task_test_inputs = [ex['input'] for ex in data[k1]['test']]
                        new_test_outputs = generate_new_output(new_def, task_test_inputs, new_key)
                        if not new_test_outputs:
                            # print('no new test outputs')
                            continue

                        new_data[f'{k1}_{k2}'] = {
                            'train': [{'input': i, 'output': o} for i, o in zip(task_train_inputs, new_train_outputs)],
                            'test': [{'input': i, 'output': o} for i, o in zip(task_test_inputs, new_test_outputs)]
                        }

                        solve_file.write(new_def)
                        solve_file.write('\n\n')

                print(len(new_data))
                data_file.write(json.dumps(new_data))
            except Exception as e:
                # print(e)
                pass


def only_create_data(solver_filename, data_filename):
    definitions = []
    with open(solver_filename, 'r') as f:
        definitions = f.read().split('\n\n')
    if len(definitions) == 0:
        print("didn't read any solvers")
        return

    data = get_data(train=True)
    new_data = {}

    n = len(definitions)
    
    for defn in tqdm(definitions, total=n):
        solvername = defn.split('\n')[0].split('(')[0].replace('def ', '')
        full_key = solvername.replace('solve_', '')
        k1 = full_key.split('_')[0]

        train_inputs = [ex['input'] for ex in data[k1]['train']]
        train_outputs = generate_new_output(defn, train_inputs, solvername)
        if train_outputs is None:
            print('no new train outputs')
            continue

        test_inputs = [ex['input'] for ex in data[k1]['test']]
        test_outputs = generate_new_output(defn, test_inputs, solvername)
        if test_outputs is None:
            print('no new test outputs')
            continue 

        new_data[full_key] = {
            'train': [{'input': i, 'output': o} for i, o in zip(train_inputs, train_outputs)],
            'test': [{'input': i, 'output': o} for i, o in zip(test_inputs, test_outputs)]
        }
    
    print(len(new_data))
    with open(data_filename, 'w') as f:
        f.write(json.dumps(new_data))


# combine_3()
only_create_data('solvers3.py', 'data3.json')

# with open('solvers3.py', 'r') as f:
#     definitions = f.read().split('\n\n')
#     defn = definitions[0]
#     solvername = defn.split('\n')[0].split('(')[0].replace('def ', '')
#     full_key = solvername.replace('solve_', '')
#     k1 = full_key.split('_')[0]

#     print(solvername, full_key, k1)


    
