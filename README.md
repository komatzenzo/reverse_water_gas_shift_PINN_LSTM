# reverse water gas shift PINN
# rWGS z-PINN / Digital Twin  First-Principles + Variables

This repository contains a **physics-informed neural network (PINN)** prototype for a **1D axial fixed-bed rWGS reactor** (plug-flow in \(z\)), designed as a starting point for:
- **Axial temperature curve prediction** \(T(z)\) (and optionally species profiles \(F_i(z)\))
- **Deactivation tracking** \(a\) (coking/poisoning) inferred from **outlet \(X_{\mathrm{CO_2}}\) and \(S_{\mathrm{CO}}\)** only
- **Transfer learning** from legacy steady-state datasets
- **Parameter identification** of kinetics \((k_0, E)\) with Monte-Carlo/bayesian sampling and PINN residual matching

> Scope: steady-state **per run**. “Time dependence” is represented via **space time / residence time** changes through **GHSV / flow**, i.e. each experimental run is treated as a separate steady operating point.

---

## 1) Reactor & Operating Space

### Geometry
- Reactor length:  
  \[
  L = 0.20\ \text{m} \quad (20\ \text{cm})
  \]
- Axial sensor positions (example for 5 points):  
  \[
  z \in \{0,\ 0.05,\ 0.10,\ 0.15,\ 0.20\}\ \text{m}
  \]

### Typical operating range (per run)
- Temperature: \(T_{\text{in}} \in [823, 1223]\ \text{K}\) (550–950 °C)
- Pressure: \(p \in [1, 10]\ \text{bar(a)}\) (example default: 5 bar(a))
- Inlet ratio: \(\mathrm{H_2/CO_2}=3\)
  \[
  y_{\mathrm{CO_2,in}}=0.25,\quad y_{\mathrm{H_2,in}}=0.75
  \]

### Catalyst loading (example lab-scale)
- Catalyst mass per length (lumped):  
  \[
  W' = 0.10\ \text{kg}_{cat}\,\text{m}^{-1}
  \]
- Total catalyst mass:  
  \[
  W = W' L = 0.02\ \text{kg}_{cat}
  \]

---

## 2) Reaction System

### Main reaction (rWGS)
\[
\mathrm{CO_2 + H_2 \rightleftharpoons CO + H_2O}
\]

### Stoichiometry (molar rates)
For reaction rate \(r\) (per catalyst mass):
- \(\nu_{\mathrm{CO_2}}=-1\)
- \(\nu_{\mathrm{H_2}}=-1\)
- \(\nu_{\mathrm{CO}}=+1\)
- \(\nu_{\mathrm{H_2O}}=+1\)

### Thermochemistry (default)
- Endothermic reaction enthalpy:
  \[
  \Delta H_{\mathrm{rWGS}} = +41\,000\ \text{J mol}^{-1}
  \]

---

## 3) Variables & Definitions (per run)

### Axial coordinate
- \(z\) [m], \(z \in [0, L]\)

### State variables (recommended minimal set)
\[
\mathbf{x}(z)=\left[T, F_{\mathrm{CO_2}},F_{\mathrm{H_2}},F_{\mathrm{CO}},F_{\mathrm{H_2O}}\right]
\]
- \(T(z)\) [K] axial gas temperature  
- \(F_i(z)\) [mol/s] molar flow of species \(i\)
- Total molar flow:
  \[
  F_T(z)=\sum_i F_i(z)
  \]
- Mole fractions:
  \[
  y_i(z)=\frac{F_i(z)}{F_T(z)}
  \]
- Partial pressures:
  \[
  p_i(z)=y_i(z)\,p
  \]
  with \(p\) as total pressure (use Pa in kinetics; \(p[\text{Pa}] = p[\text{bar}]\cdot 10^5\)).

### Inputs (typical DoE features)
- \(T_{\text{in}}\) / setpoint \(T_{set}\)
- \(p\) [bar(a)]
- GHSV or flow (affects \(F_T\))
- Inlet ratio \(H_2/CO_2\)

### Outputs used for deactivation detection (experiment)
- CO\(_2\) conversion:
  \[
  X_{\mathrm{CO_2}}=\frac{F_{\mathrm{CO_2,in}}-F_{\mathrm{CO_2,out}}}{F_{\mathrm{CO_2,in}}}
  \]
- CO selectivity (for rWGS, carbon basis; if only CO/CH4 measured adapt):
  \[
  S_{\mathrm{CO}}=\frac{F_{\mathrm{CO,out}}-F_{\mathrm{CO,in}}}{(F_{\mathrm{CO,out}}-F_{\mathrm{CO,in}})+(F_{\mathrm{CH_4,out}}-F_{\mathrm{CH_4,in}})}
  \]
  If CH\(_4\) not available (or negligible), a simplified proxy is:
  \[
  S_{\mathrm{CO}} \approx \frac{F_{\mathrm{CO,out}}-F_{\mathrm{CO,in}}}{F_{\mathrm{CO_2,in}}-F_{\mathrm{CO_2,out}}}
  \]
  (only valid if CO is the dominant carbon product).

---

## 4) First-Principles Model (1D Plug Flow)

### 4.1 Mass balances (per length)
Let \(r(z)\) be the reaction rate per catalyst mass \([\text{mol s}^{-1}\text{kg}_{cat}^{-1}]\).
Convert to per length using \(W'\) \([\text{kg}_{cat}\,\text{m}^{-1}]\):

\[
\frac{dF_i}{dz} = \nu_i\, r(z)\,W'
\]

Explicitly:
\[
\begin{aligned}
\frac{dF_{\mathrm{CO_2}}}{dz} &= -rW' \\
\frac{dF_{\mathrm{H_2}}}{dz}  &= -rW' \\
\frac{dF_{\mathrm{CO}}}{dz}   &= +rW' \\
\frac{dF_{\mathrm{H_2O}}}{dz} &= +rW'
\end{aligned}
\]

### 4.2 Energy balance (gas phase, lumped heat transfer)
\[
\frac{dT}{dz} =
\frac{-\Delta H_{\mathrm{rWGS}}\ rW' + UA'\,(T_w - T)}
{F_T\, C_{p,\text{mix}}}
\]

Where:
- \(UA'\) [W K\(^{-1}\) m\(^{-1}\)] effective heat transfer per length
- \(T_w\) [K] wall/furnace temperature (often \(T_{set}\))
- \(C_{p,\text{mix}}\) [J mol\(^{-1}\) K\(^{-1}\)] mixture heat capacity (toy constant default)

**Default toy values**
- \(C_{p,\text{mix}} = 35\ \text{J mol}^{-1}\text{K}^{-1}\)
- \(UA' = 15\ \text{W K}^{-1}\text{m}^{-1}\)

---

## 5) Kinetics (Arrhenius + Driving Force)

### 5.1 Arrhenius law
\[
k(T)=k_0\exp\left(-\frac{E}{RT}\right)
\]
- \(R=8.314\ \text{J mol}^{-1}\text{K}^{-1}\)
- \(k_0\), \(E\) to be identified or sampled

### 5.2 Equilibrium term (simple smooth surrogate)
Use a 2-parameter approximation:
\[
\ln K_{eq}(T) = A + \frac{B}{T}
\]
Typical toy defaults:
- \(A=-3.0\)
- \(B=4000\ \text{K}\)

### 5.3 Reaction rate (driving force form)
\[
r = a\ k(T)\left(p_{\mathrm{CO_2}}p_{\mathrm{H_2}}-\frac{p_{\mathrm{CO}}p_{\mathrm{H_2O}}}{K_{eq}(T)}\right)
\]

- \(a \in (0,1]\) is **activity** (deactivation factor)
- \(p_i\) in Pa (recommended for consistency)

**Toy but realistic parameter guesses**
- \(E = 95\,000\ \text{J mol}^{-1}\)
- \(k_0 = 1.0\times 10^{-4}\ \text{mol}\,\text{s}^{-1}\,\text{Pa}^{-2}\,\text{kg}_{cat}^{-1}\)

> These values are for stable prototyping. For publications, kinetics must be taken from a validated source or fitted to your data.

---

## 6) Deactivation Model (for synthetic testing)

To test closed-loop detection/control, define activity per run \(n\):

### Exponential decay (typical)
\[
a(n) = \exp\left(-\frac{n}{\tau}\right)
\]
Example: \(\tau=60\) runs

### Or a bounded linear trend
\[
a(n)=\max(0.6,\ 1-0.005n)
\]

In the real use case, \(a\) is **not known** and must be inferred from deviations in \(X_{\mathrm{CO_2}}\) and \(S_{\mathrm{CO}}\) (and optionally temperature profiles).

---

## 7) Why predicting \(T(z)\) is necessary (for deactivation inference)

Deactivation is often confounded with thermal effects:
- rWGS is **endothermic**: a drop in heat transfer / temperature can mimic lower activity.
- Kinetics depends strongly on temperature:
  \[
  k(T)\propto \exp(-E/RT)
  \]
Thus, inferring \(a\) purely from \(X_{\mathrm{CO_2}}\) and \(S_{\mathrm{CO}}\) benefits from a model that also reconstructs \(T(z)\). A z-PINN provides this by enforcing the energy balance while fitting sparse axial temperature sensors.

---

## 8) PINN Formulation (z-PINN)

### 8.1 Neural surrogate
Define a network \(f_\theta\) predicting states:
\[
\hat{\mathbf{x}}(z;\theta) = [\hat{T}, \hat{F}_{\mathrm{CO_2}},\hat{F}_{\mathrm{H_2}},\hat{F}_{\mathrm{CO}},\hat{F}_{\mathrm{H_2O}}]
\]

Option A (temperature-only PINN):
\[
\hat{T}(z;\theta)\ \text{only}
\]
and use measured outlet compositions for mass closure.

Option B (full state PINN):
predict \(T(z)\) and \(F_i(z)\) jointly.

### 8.2 Physics residuals
Mass residuals:
\[
\mathcal{R}_{F_i}(z)=\frac{d\hat{F_i}}{dz} - \nu_i\,\hat{r}(\hat{\mathbf{x}})\,W'
\]
Energy residual:
\[
\mathcal{R}_T(z)=\frac{d\hat{T}}{dz} -
\frac{-\Delta H\,\hat{r}W' + UA'(T_w - \hat{T})}{\hat{F_T} C_{p,\text{mix}}}
\]

### 8.3 Loss function (single run)
Data loss (temperatures at sensor positions \(z_k\)):
\[
\mathcal{L}_{data}=\sum_k \left\|\hat{T}(z_k)-T_{meas}(z_k)\right\|^2
\]
Physics loss (collocation points \(z_c\)):
\[
\mathcal{L}_{phys}=\sum_{z_c}\left(\|\mathcal{R}_T(z_c)\|^2+\sum_i\|\mathcal{R}_{F_i}(z_c)\|^2\right)
\]
Boundary loss (inlet conditions):
\[
\mathcal{L}_{bc}=\|\hat{T}(0)-T_{in}\|^2+\sum_i\|\hat{F_i}(0)-F_{i,in}\|^2
\]

Total:
\[
\mathcal{L}=\lambda_{data}\mathcal{L}_{data}+\lambda_{phys}\mathcal{L}_{phys}+\lambda_{bc}\mathcal{L}_{bc}
\]

---

## 9) Estimating \(k_0\) and \(E\) (Monte-Carlo + PINN constraint)

### Approach
1. Treat \(k_0\), \(E\) (and possibly \(UA'\), \(C_p\)) as **unknown global parameters**.
2. Sample candidate values:
   \[
   E \sim \mathcal{U}(70,140)\,\text{kJ/mol},\quad
   \log_{10}k_0 \sim \mathcal{U}(-8,-2)
   \]
3. For each sample, run the PINN training (or a fast residual evaluation if using a fixed surrogate).
4. Keep parameter sets minimizing:
   - outlet mismatch in \(X_{\mathrm{CO_2}}\), \(S_{\mathrm{CO}}\) (if available)
   - physics residual norms \(\mathcal{L}_{phys}\)

### Practical note
For efficiency:
- pretrain \(\theta\) on legacy steady-state data (transfer learning),
- then only fine-tune a small number of epochs per MC sample,
- or use an outer loop optimizing \(k_0,E\) with gradient-based methods if differentiable.

---

## 10) Explainability & Control linkage (SHAP / CCD)

Once a surrogate exists (PINN or purely data-driven), you can:
- compute SHAP feature attributions on outputs \((X_{\mathrm{CO_2}}, S_{\mathrm{CO}}, Y=X\cdot S)\)
- interpret which inputs (T, p, ratio, GHSV/flow) drive performance and deactivation signals
- design **CCD / DoE** focused on influential factors
- derive **control priorities / weighting** for an MPC or rule-based controller

---

## 11) Recommended Next Steps
- Start with **temperature-only z-PINN** using 5 axial sensors.
- Add outlet \(X_{\mathrm{CO_2}}\) constraint via mass balance closure.
- Introduce deactivation \(a\) as a **latent per-run parameter**, constrained by outlet conversion/selectivity.
- Finally expand to full state \(F_i(z)\) if you have reliable gas composition signals.

---

## License / Notes
This is a prototype specification (toy-consistent). For publication-grade work, replace kinetics and equilibrium models with validated sources and calibrate units consistently to your reactor definition.
