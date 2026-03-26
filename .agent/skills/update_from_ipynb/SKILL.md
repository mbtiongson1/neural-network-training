---
name: Update Python Architecture from Ipynb
description: Workflow to update a modular Python architecture based on a Jupyter notebook source of truth
---

# Update Python Architecture from Ipynb

This skill describes the workflow for extracting logic from a Jupyter notebook (`.ipynb`) and refactoring/updating a modular python architecture (e.g., a `python/` directory with multiple modules) to synchronize perfectly with the notebook as the source of truth.

## Execution Steps

1. **Extract Python Code from Notebook:**
   Use the `jupyter nbconvert --to script <notebook_path>` command to convert the target `.ipynb` file into a `.py` script. This makes it easier to read and parse the notebook's logic. (e.g. `jupyter nbconvert --to script submission/main.ipynb`)

2. **Analyze the Source of Truth:**
   Read the newly extracted `.py` script and any other auxiliary scripts (e.g., scoring scripts). Understand the latest configurations, hyper-parameters, internal loops, and training phases.

3. **Map the Changes to the Python Modules:**
   Review the current state of the python modules (e.g., `config.py`, `activations.py`, `network.py`, `utils.py`, `main.py`). Match the updated notebook logic to their respective modular files.

4. **Update the Python Architecture:**
   Use file manipulation tools to overwrite or carefully replace the code inside the modular `.py` files. 
   - Extract constants and dictionaries into `config.py`.
   - Extract math or helper functions into files like `activations.py` or `utils.py`.
   - Extract class definitions into files like `network.py`.
   - Consolidate the notebook's sequential training execution into `main.py`.

5. **Update Documentation / Deployment Instructions:**
   Modify the `README.md` to indicate how the python workflow should be deployed or run natively. Ensure instructions guide the user to navigate to the python directory and run the standalone pipeline.
