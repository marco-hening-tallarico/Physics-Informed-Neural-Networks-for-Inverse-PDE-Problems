# Physics-Informed Neural Networks for Inverse PDE Problems

A collection of Jupyter notebooks demonstrating Physics-Informed Neural Networks (PINNs) applied to inverse problems for partial differential equations (PDEs).  
This repository focuses on notebook-based experiments, visualizations, and reproducible example pipelines for inferring unknown coefficients, source terms, or parameters of PDEs from data while enforcing physics constraints.

Repository language composition: Jupyter Notebook (100%)

---

## Contents

- notebooks/ (root)
  - Example notebooks showing PINN formulations for 1D/2D inverse problems, data generation, training, and evaluation.
- data/ (optional)
  - Synthetic datasets or scripts to generate them.
- environment.yml / requirements.txt (recommended)
  - Specification of Python packages needed to run the notebooks.
- LICENSE
- README.md (this file)

If your repo currently holds only notebooks, consider adding a `requirements.txt` or `environment.yml` and a small `scripts/` folder for programmatic reproducibility.

---

## Quick start

Choose one of the options below depending on how you prefer to run notebooks.

Run locally
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   venv\Scripts\activate      # Windows
   ```
2. Install dependencies (if a `requirements.txt` exists):
   ```bash
   pip install -r requirements.txt
   ```
   Or create a conda env:
   ```bash
   conda env create -f environment.yml
   conda activate pinn-inverse
   ```
3. Launch Jupyter Lab / Notebook:
   ```bash
   jupyter lab
   ```
4. Open the notebooks and run cells interactively.

Run on Google Colab
- Open the notebook in GitHub and click “Open in Colab” (you can add Colab badges or links per notebook). If notebooks access data, provide public URLs or include small sample data in `data/`.

Run with Binder
- Add a `requirements.txt` or `environment.yml` and a `postBuild` script to enable Binder launch:
  - Binder badge:
    ```
    https://mybinder.org/v2/gh/<owner>/<repo>/HEAD
    ```

Hardware
- CPU is sufficient for small examples. For larger PINNs or faster experiments, use a machine with an NVIDIA GPU and configure the notebooks to use PyTorch/TensorFlow GPU devices.

---

## Notebooks overview (suggested)

- inverse_1d.ipynb — 1D inverse problem: infer coefficient or source term from noisy observations.
- inverse_2d.ipynb — 2D inverse example (conductivity, diffusion, etc.).
- data_generation.ipynb — scripts for synthetic data generation and measurement noise models.
- training_utils.ipynb — examples of network architectures, loss weighting strategies, and training loops.
- evaluation_and_vis.ipynb — metrics, error analysis, and visualization of inferred fields.

Add or update README sections in each notebook explaining purpose, inputs, expected runtime, and required cell order.

---

## Recommended dependencies

Typical packages used in PINN notebooks:

- Python 3.8+
- numpy, scipy, matplotlib, pandas
- JupyterLab / notebook
- PyTorch (or TensorFlow) — pick one and document it
- tqdm, seaborn, scikit-learn, yaml

Example `requirements.txt` snippet:
```text
numpy
scipy
matplotlib
pandas
jupyterlab
torch    # or tensorflow
tqdm
seaborn
scikit-learn
pyyaml
```

---

## Reproducibility tips

- Add seeds in notebooks for `numpy`, `random`, and `torch` (or `tf`) to make results repeatable.
- Save training artifacts (checkpoints, figures) into an `experiments/` folder with a small manifest (config + git commit hash).
- Keep one notebook that demonstrates end-to-end data → train → evaluate with a fixed small configuration that runs quickly on CPU.

---

## Suggestions for improving this repo

- Add `requirements.txt` or `environment.yml` for easy environment setup.
- Add a `notebooks/README.md` summarizing each notebook and recommended run order.
- Provide a small sample dataset or a `data_generation.ipynb` so users can reproduce results offline.
- Add a lightweight `src/` folder with reusable functions (data loaders, loss definitions, network builders) and import them from notebooks — this helps transition to scripted experiments.
- Add CI checks (optional) that run the quick smoke notebook(s) via nbval or convert them to minimal pytest-compatible unit tests.

---

## Citation

If you use these notebooks in research, please cite this repository:
```
@misc{marco-hening-tallarico-2026-pinn-inverse,
  title = {Physics-Informed Neural Networks for Inverse PDE Problems},
  author = {marco-hening-tallarico},
  year = {2026},
  howpublished = {GitHub repository},
  note = {https://github.com/marco-hening-tallarico/Physics-Informed-Neural-Networks-for-Inverse-PDE-Problems}
}
```

---

## Contributing

Contributions are welcome. Suggested workflow:
1. Open an issue describing the feature, improvement, or bug.
2. Create a branch named `feat/<short-name>` or `fix/<issue-number>`.
3. Add tests or a smoke notebook showing the fix.
4. Submit a pull request explaining changes and linking the issue.

---

## License

Add an appropriate LICENSE file (e.g., MIT) to make reuse clear.

---

## Contact

Maintainer: marco-hening-tallarico  
Repository: https://github.com/marco-hening-tallarico/Physics-Informed-Neural-Networks-for-Inverse-PDE-Problems
