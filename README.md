# Physics-Informed Machine Learning for a Damped Harmonic Oscillator

A small scientific machine-learning project comparing a conventional
neural network with a physics-informed neural network (PINN) for
modelling the displacement of a damped harmonic oscillator.

The project investigates whether incorporating a governing differential
equation can improve predictions when observations are sparse or noisy.

## Project overview

Many machine-learning models learn patterns from examples without
explicitly using the physical laws that govern the system. A
physics-informed neural network adds a physics-based penalty to its
training objective, encouraging its predictions to satisfy a known
differential equation.

This project uses a simulated damped harmonic oscillator as a controlled
test case. It compares:

-   A **data-only neural network**, trained on observed displacement
    data and the initial conditions.
-   A **physics-informed neural network (PINN)**, trained on observed
    displacement data, the initial conditions, and the governing
    differential equation.

The experiments explore test performance, physics-loss weighting,
reduced observation counts, and noisy observations.

## Governing physics

The oscillator is described by

\[ m`\frac{d^2x}{dt^2}`{=tex} + c`\frac{dx}{dt}`{=tex} + kx = 0, \]

where:

-   (x(t)) is displacement,
-   \(m\) is mass,
-   \(c\) is the damping coefficient,
-   \(k\) is the spring stiffness.

The simulation uses:

  Parameter                              Value
  ---------------------------------- ---------
  Mass, (m)                                1.0
  Damping coefficient, (c)                 0.5
  Spring stiffness, (k)                    4.0
  Initial displacement, (x(0))             1.0
  Initial velocity, (x'(0))                0.0
  Simulation interval                  0--10 s
  Number of simulated observations       1,000

The reference trajectory is generated numerically and saved as
`data/oscillator_data.csv`.

## How the PINN works

The neural network takes time (t) as input and predicts displacement
(x\_`\theta`{=tex}(t)). Automatic differentiation is used to calculate
its first and second derivatives with respect to time.

The physics residual is

\[ R\_`\theta`{=tex}(t) = m`\frac{d^2x_\theta}{dt^2}`{=tex}
+c`\frac{dx_\theta}{dt}`{=tex} +kx\_`\theta`{=tex}(t). \]

A perfect solution to the governing equation has a residual of zero. The
PINN penalizes the squared residual at collocation points sampled across
the time domain.

The training objective is

\[ `\mathcal{L}`{=tex} = `\mathcal{L}`{=tex}*{`\mathrm{data}`{=tex}}
+`\lambda`{=tex}*{`\mathrm{physics}`{=tex}}`\mathcal{L}`{=tex}*{`\mathrm{physics}`{=tex}}
+`\mathcal{L}`{=tex}*{`\mathrm{initial}`{=tex}}, \]

where:

-   (`\mathcal{L}`{=tex}\_{`\mathrm{data}`{=tex}}) measures the
    difference between predicted and observed displacement.
-   (`\mathcal{L}`{=tex}\_{`\mathrm{physics}`{=tex}}) penalizes
    violations of the differential equation.
-   (`\mathcal{L}`{=tex}\_{`\mathrm{initial}`{=tex}}) enforces the
    initial displacement and velocity.
-   (`\lambda`{=tex}\_{`\mathrm{physics}`{=tex}}) controls the strength
    of the physics penalty.

The experiments use a fully connected neural network with three hidden
layers of 32 neurons and `tanh` activations. The main PINN experiments
use a physics-loss weight of 100 unless otherwise specified.

## Main validation result

In the repeated-run validation experiment, both models were trained on
observations from approximately the first 80% of the time interval and
evaluated on the remaining 20% (approximately 8--10 seconds). The
physics residual was evaluated across the full 0--10 second domain.

Mean test RMSE over three random seeds:

  -----------------------------------------------------------------------
        Fraction of         Number of Data-only NN RMSE         PINN RMSE
           training      observations                   
       observations                                     
  ----------------- ----------------- ----------------- -----------------
               100%               800            0.5407            0.0781

                50%               400            0.5496            0.0790

                25%               200            0.5050            0.0781

                10%                80            0.6499            0.0780
  -----------------------------------------------------------------------

For these runs, the PINN's mean RMSE was approximately 84--88% lower
than the data-only neural network's mean RMSE, depending on the
training-data fraction.

The PINN's RMSE was also more consistent across the three seeds. The
data-only network's performance varied substantially between
initializations.

**Interpretation:** In this simulated oscillator experiment, the physics
constraint substantially improved reconstruction on the held-out
interval, including when fewer observed data points were used.

### Important qualification

This is an encouraging result for this specific simulated system, not
proof that PINNs always outperform conventional neural networks. The
PINN receives a strong advantage from knowing the correct governing
equation and enforcing its residual across the full time domain,
including the held-out interval. This is therefore a **physics-informed
reconstruction/generalization experiment**, not a completely blind
future forecast.

The baseline also uses the initial displacement and velocity
constraints. It is more precise to describe it as a neural network
trained on data and initial conditions **without a physics-residual
loss**, rather than a model with no physical information at all.

The experiment currently uses only three random seeds. More seeds,
alternative model architectures, and additional physical systems would
be needed to establish broader conclusions.

## Experiments

The project includes the following investigations:

1.  **Baseline comparison** --- compares a data-only neural network with
    a PINN.
2.  **Physics-loss weight** --- examines how different values of
    (`\lambda`{=tex}\_{`\mathrm{physics}`{=tex}}) affect performance.
3.  **Sparse observations** --- compares performance using 100%, 50%,
    25%, and 10% of the available training observations.
4.  **Noisy observations** --- tests models with different levels of
    added noise in training observations.
5.  **Repeated-run validation** --- repeats the sparse-observation
    comparison using three random seeds and summarizes mean performance
    and standard deviation.
6.  **Diagnostics** --- plots predicted trajectories, zooms in on the
    held-out interval, and examines the PINN's governing-equation
    residual.

Results are saved as CSV files and figures in the `results/` directory.

## Repository structure

The project is intended to follow this structure. Your exact filenames
may differ slightly depending on which scripts you have created.

``` text
physics-informed-ml/
├── data/
│   └── oscillator_data.csv
├── results/
│   ├── validation_trials.csv
│   ├── validation_summary.csv
│   ├── validation_rmse.png
│   ├── diagnostic_predictions.csv
│   ├── diagnostic_metrics.csv
│   ├── diagnostic_predictions.png
│   ├── diagnostic_test_interval.png
│   └── diagnostic_physics_residual.png
├── main.py
├── ml_baseline.py
├── nn_baseline.py
├── pinn_model.py
├── pinn_experiment.py
├── pinn_noise_experiment.py
├── pinn_sparse_experiment.py
├── pinn_validation.py
├── pinn_diagnostics.py
├── requirements.txt
└── README.md
```

Keep your virtual environment (for example, `myenv/`) separate from the
source files. Do not commit the environment folder to GitHub.

## Installation

You will need Python 3.10 or later and the packages used by your
scripts.

From the project root, create and activate a virtual environment.

### Windows PowerShell

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install dependencies:

``` powershell
pip install numpy pandas matplotlib scipy scikit-learn torch
```

If PowerShell blocks environment activation, you can run the
environment's Python executable directly or follow the Python
documentation for your system's execution policy.

## Running the project

Run commands from the project root---the directory containing `data/`,
`results/`, and the Python scripts.

Generate the reference dataset and initial plot:

``` powershell
python main.py
```

Run the experiments that are present in your local copy, for example:

``` powershell
python pinn_experiment.py
python pinn_noise_experiment.py
python pinn_sparse_experiment.py
python pinn_validation.py
python pinn_diagnostics.py
```

Some experiments may take several minutes because the models are trained
for thousands of epochs. Each script may overwrite files with the same
output names, so keep copies of important results if you modify the
experiments.

## Metrics

The project reports:

-   **Mean Absolute Error (MAE):** average absolute prediction error.
-   **Mean Squared Error (MSE):** average squared prediction error.
-   **Root Mean Squared Error (RMSE):** square root of MSE, which gives
    greater weight to larger errors.

For reference values (x_i) and predictions (`\hat{x}`{=tex}\_i),

\[
`\mathrm{MAE}`{=tex}=`\frac{1}{N}`{=tex}`\sum`{=tex}\_{i=1}\^{N}\|x_i-`\hat{x}`{=tex}\_i\|,
\]

\[ `\mathrm{RMSE}`{=tex}=
`\sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i-\hat{x}_i)^2}`{=tex}. \]

Lower values indicate smaller errors on the evaluated dataset.
Displacement is measured in the simulation's displacement units.

## Limitations and future work

Current limitations include:

-   The reference data are simulated, rather than measured from a
    physical experiment.
-   The model is evaluated on one relatively simple, known differential
    equation.
-   The governing equation is supplied to the PINN, so the result
    depends on the correctness of that physical knowledge.
-   Physics collocation points cover the held-out interval, which makes
    this reconstruction setup different from forecasting with no
    constraints in the future interval.
-   Only three seeds are used in the repeated-run validation.
-   Model architecture, loss scaling, collocation-point selection, and
    optimization settings can affect results.

Possible extensions:

-   Increase the number of random seeds and report confidence intervals.
-   Compare with a numerical solver and a conventional forecasting
    baseline.
-   Normalize time and displacement, and investigate loss-term scaling.
-   Test sensitivity to incorrect physical parameters and imperfect
    governing equations.
-   Evaluate a truly prospective forecasting setup with physics
    constraints restricted to the training domain, or clearly compare
    several collocation strategies.
-   Apply the method to a more complex system or to measured
    experimental data.

## Skills demonstrated

-   Python scientific computing
-   Numerical simulation of differential equations
-   Neural networks with PyTorch
-   Automatic differentiation
-   Physics-informed machine learning
-   Model evaluation and error metrics
-   Controlled experiments and repeated-run validation
-   Data visualization and reproducible research workflows

## Reproducibility

Random seeds are set in the main experiment scripts where implemented.
Results can vary with package versions, hardware, and optimization
behavior. For a reproducible submission, record the Python and package
versions, keep the experiment scripts and configuration values under
version control, and regenerate the reported tables from the saved CSV
outputs.

## Author

**Rabbit**\
Physics \| Applied Mathematics \| Scientific Computing \| Machine
Learning

This is an independent learning and portfolio project exploring the use
of physical knowledge in machine learning.
