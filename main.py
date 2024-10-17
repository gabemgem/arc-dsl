import os
import json
import inspect

import numpy as np
import tqdm

import arc_types
import constants
import dsl
import tests
import solvers

import multiprocessing
from joblib import Parallel, delayed
from distance import distance



def tuple_to_list(t):
    # return [tuple_to_list(e) if isinstance(e, tuple) else e for e in t]
    return np.array(t).tolist()

def list_to_tuple(l):
    return tuple(tuple(r) for r in l)

def get_data_old(train=True):
    path = f'../data/{"training" if train else "evaluation"}'
    data = {}
    for fn in os.listdir(path):
        with open(f'{path}/{fn}') as f:
            data[fn.split(".")[0]] = json.load(f)
    ast = lambda g: tuple(tuple(r) for r in g)
    data_key = "training" if train else "evaluation"
    return {
        'train': {k: [{
            'input': ast(e['input']),
            'output': ast(e['output']),
        } for e in v['train']] for k, v in data[data_key].items()},
        'test': {k: [{
            'input': ast(e['input']),
            'output': ast(e['output']),
        } for e in v['test']] for k, v in data[data_key].items()}
    }

def get_data(train=True):
    path = f'../data/{"training" if train else "evaluation"}'
    data = {}
    # for fn in os.listdir(path):
    #     with open(f'{path}/{fn}') as f:
    #         data[fn.split(".")[0]] = json.load(f)
    challenges_f = '../data/arc-agi_training_challenges.json'
    solutions_f = '../data/arc-agi_training_solutions.json'
    solutions = {}
    with open(challenges_f) as f:
        data = json.load(f)
    with open(solutions_f) as f:
        solutions = json.load(f)

    for key, outputs in solutions.items():
        for i in range(len(data[key]['test'])):
            data[key]['test'][i]['output'] = outputs[i]

    """
    data looks something like:
    {
    "key": {
        "test": [{"input": [], "output": []}, ...],
        "train": [{"input": [], "output": []}, ...]
    },
    }
    """
    return data


def get_functions(path):
    """ returns a list of available functions """
    with open(path, 'r') as f:
        code = f.read()
    functions = []
    for row in code.split('\n'):
        if row.startswith('def '):
            function = row.split('def ')[1].split('(')[0]
            functions.append(function)
    return functions


def run_dsl_tests(dsl_module, test_module):
    """ test DSL primitives """
    dsl_functions = get_functions(dsl_module.__file__)
    test_functions = get_functions(test_module.__file__)
    expected = set([f'test_{f}' for f in dsl_functions])
    assert set(test_functions) == expected
    for fun in test_functions:
        getattr(test_module, fun)()


def test_solvers_formatting(solvers_module, dsl_module):
    """ tests the implementd solvers for formatting """
    with open('constants.py', 'r') as f:
        constants = [c.split(' = ')[0] for c in f.readlines() if ' = ' in c]
    definitions = {
        function: inspect.getsource(getattr(solvers_module, function)) \
            for function in get_functions(solvers_module.__file__)
    }
    dsl_interface = get_functions(dsl_module.__file__)
    n_correct = 0
    n = len(definitions)
    for key, definition in definitions.items():
        try:
            lines = definition.split('\n')
            assert lines[0] == f'def {key}(I):'
            assert lines[-1] == ''
            assert lines[-2] == '    return O'
            variables = set()
            calls = set()
            for line in lines[1:-2]:
                variable, call = line.lstrip().split(' = ')
                function, args = call.split('(')
                assert variable not in dsl_interface
                assert variable not in variables
                assert call not in calls
                variables.add(variable)
                calls.add(call)
                assert function in dsl_interface or function in variables
                assert args[-1] == ')'
                args = [args[:-1]] if ',' not in args else args[:-1].split(', ')
                for arg in args:
                    assert any([
                        arg in variables, arg in dsl_interface,
                        arg in constants, arg == 'I'
                    ])
            for v in variables:
                assert sum([
                    definition.count(vs) for vs in [
                        f'({v})', f'({v}, ', f', {v})',
                        f', {v}, ', f' {v} = ', f' {v}('
                    ]
                ]) > 1 or v == 'O'
            n_correct += 1
        except:
            print(definition)
            pass
    print(f'{n_correct} out of {n} solvers formatted correctly.')


def test_solvers_correctness(data, solvers_module):
    """ tests the implemented solvers for correctness """
    n_correct = 0
    n = len(data)
    for key in tqdm.tqdm(data.keys(), total=n):
        task = data[key]['train'] + data[key]['test']
        try:
            if not hasattr(solvers_module, f'solve_{key}'):
                continue
            solver = getattr(solvers_module, f'solve_{key}')
            for ex in task:
                input = list_to_tuple(ex['input'])
                output = list_to_tuple(ex['output'])
                attempt = solver(input)
                assert attempt == output
            n_correct += 1
        except Exception as e:
            print(e)
            # input = list_to_tuple(ex['input'])
            # output = list_to_tuple(ex['output'])
            # attempt = solver(input)
            print(key)
            # print(input)
            # print(attempt)
            # print(output)
            # print(attempt == output)
            # print(json.dumps(ex['input']))
            # print(json.dumps(tuple_to_list(solver(ex['input']))))
            # print(json.dumps(ex["output"]))
            break
            pass
    print(f'{n_correct} out of {n} tasks solved correctly.')

def test_solvers_on_evaluation_set(data, solvers_module, use_full_cpu=True):
    """ tries all of the existing solvers on the evaluation dataset """
    n_correct = 0
    n = len(data["train"])

    num_cores = multiprocessing.cpu_count()
    if not use_full_cpu and num_cores > 1:
        num_cores -= 1

    keys = data['train'].keys()
    tasks = tqdm.tqdm([data['train'][key] + data['test'][key] for key in keys], total=n)
    outputs = Parallel(n_jobs=num_cores)(delayed(test_solvers_on_task)(task, solvers_module) for task in tasks)

    n_correct = sum(o_dist == 0 for (_, o_dist) in outputs)
    improvements = [o_dist - i_dist for (i_dist, o_dist) in outputs]
    print(f'{n_correct} out of {n} tasks solved correctly.')
    print(f'Average improvement: {np.mean(improvements)}')

def test_solvers_on_task(task, solvers_module):
    initial_distance = np.mean([distance(ex['input'], ex['output']) for ex in task])
    min_distance = initial_distance
    solvers = get_functions(solvers_module.__file__)
    for solvername in solvers:
        try:
            solver = getattr(solvers_module, solvername)
            results = [distance(solver(ex['input']), ex['output']) for ex in task]
            result_distance = np.mean(results)
            if result_distance == 0:
                print(solvername)
                return (initial_distance, 0)
            elif result_distance < min_distance:
                min_distance = result_distance
        except:
            pass
    return (initial_distance, min_distance)



def main():
    data = get_data(train=True)
    run_dsl_tests(dsl, tests)
    test_solvers_formatting(solvers, dsl)
    test_solvers_correctness(data, solvers)
    #test_solvers_on_evaluation_set(data, solvers, use_full_cpu = False)


if __name__ == '__main__':
    main()
