import solvers
# import solvers2_full
# import solvers3
import inspect
import json
from tqdm import tqdm

from main import get_data, get_functions

def create_training_set(solver_module, previous_training_set=None, input_data=None):
    data = {}
    if input_data is None:
        data = get_data(train=True)
    else:
        with open(input_data, 'r') as f:
            data = json.load(f)
    if len(data) == 0:
        print('no input data')
        return

    training_set = {}
    if previous_training_set is not None:
        with open(previous_training_set, 'r') as f:
            training_set = json.load(f)
    print(f'Original training set size: {len(training_set)}')

    definitions = []
    with open(solver_module, 'r') as f:
        definitions = f.read().split('\n\n')
    if len(definitions) == 0:
        print("didn't read any solvers")
        return

    n = len(definitions)

    for defn in tqdm(definitions, total=n):
        try:
            if "solve" not in defn:
                continue
            solvername = defn.replace('\n', '').split('(')[0].split('def ')[1]
            key = solvername.replace('solve_', '')

            if key not in data:
                continue

            stripped_defn = '\n'.join(defn.split('\n')[1:-1])

            training_set[key] = {
                "test": data[key]['test'],
                "train": data[key]['train'],
                "solver": stripped_defn
            }
        except Exception as e:
            print(e)
            pass

    print(f'New training set size: {len(training_set)}')
    with open('training_set.json', 'w') as f:
        json.dump(training_set, f)

create_training_set('solvers2_full.py', 'training_set1.json', 'data2_full.json')