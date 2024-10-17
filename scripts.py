import solvers
import solvers2
import solvers3
import inspect
import json

from main import get_data, get_functions

def create_training_set():
    data = get_data(train=True)

    training_set = {}

    definitions = {
        function: inspect.getsource(getattr(solvers, function)) \
        for function in get_functions(solvers.__file__)
    }

    for key, defn in definitions.items():
        stripped_key = key.split('solve_')[1]

        if stripped_key not in data:
            continue

        stripped_defn = '\n'.join(defn.split('\n')[1:-1])

        training_set[stripped_key] = {
            "test": data[stripped_key]['test'],
            "train": data[stripped_key]['train'],
            "solver": stripped_defn
        }

    with open('training_set.json', 'w') as f:
        json.dump(training_set, f)

create_training_set()