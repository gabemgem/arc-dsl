# Domain Specific Language for the Abstraction and Reasoning Corpus (ARC-DSL)

This repo is an extension of Michael Hodel's domain specific language for the Abstraction and Reasoning Corpus (ARC-DSL). The original github repository can be found [here](https://github.com/michaelhodel/arc-dsl).

The purpose of this extension is to generate new synthetic data for the ARC challenge by combining DSL solvers and running the original data through the combined solvers. The new data also includes the combined solvers, which can be used to train a model to predict new solvers to test inputs.

Here is an outline of the major additions to the original repo:

## SYNTHETIC.PY

This file contains the code to generate new synthetic data. The main functions are `concat_solvers()` and `generate_new_output()`. 

`concat_solvers()` takes in two solvers and combines them into a format that is consistent with the other solvers. `generate_new_output()` takes in the new solver and the data inputs from the first combined solver and generates new outputs.

By combining two or more solvers, we can generate exponentially more data than the original dataset. Of course, not all combined solvers with successfully generate new outputs, but by combining two and three solvers, we were able to generate approximately 500,000 new data points.

## DISTANCE.PY

This file defines a helper function, `distance()` that calculates the distance between two grids as a function of pixel matching, overall color matching, grid size matching, and foreground object matching. It can be used, for example, in an RL method of creating ARC solutions.