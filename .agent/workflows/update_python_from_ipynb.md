---
description: Update Python architecture based on a Jupyter Notebook source of truth
---

# Update Python Architecture from Ipynb

This workflow lays out the explicit steps to sync a modular python codebase with a Jupyter notebook that acts as the source of truth.

1. Extract the python code from the notebook using the command line: 
`jupyter nbconvert --to script <notebook_path>`
2. Analyze the extracted `.py` script, noting changes to hyper-parameters, architectures, utility classes, and the exact order of execution in the notebook blocks.
3. Review the existing modular codebase (e.g. your `python/` directory).
4. Update the modular python files by mapping the notebook script logic respectively:
   - Configuration / Constants -> `config.py`
   - Classes -> e.g. `network.py`
   - Helpers -> `activations.py`, `utils.py`
   - Step-by-step pipeline execution -> `main.py`
5. Rewrite and refine the `README.md` deployment sections to instruct running the modular python implementation natively (e.g., detailing the commands to install requirements and run `main.py`).
