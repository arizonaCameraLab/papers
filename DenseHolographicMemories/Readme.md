# Dense Holographic Associative Memories

Simulation notebooks for

> D. J. Brady and G. Nero, "Dense holographic associative memories," submitted to *Optics Express* (2026).

The paper shows that a cascade of two volume holograms around a nonlinear hidden layer, one memory neuron per stored pattern, physically evaluates the modern Hopfield (dense associative memory) retrieval map $\hat{\boldsymbol\eta} = V\,\mathrm{softmax}(\lambda K^{\top}\mathbf{x})$, and proposes a gradient-responsive recording medium with illumination-independent decay that recovers the linear $M^{-1}$ per-hologram efficiency scaling in situ. The two notebooks here generate the paper's simulation figures.

## Notebooks

| Notebook | Figure | Content |
|---|---|---|
| [`DAMShift.ipynb`](DAMShift.ipynb) | Fig. 2 | Shift-multiplexed realization of the hidden layer: shift selectivity and memory-neuron count versus hologram thickness, and recall fidelity versus page size with linear and softmax read-out. |
| [`DAMMno.ipynb`](DAMMno.ipynb) | Fig. 3 | Growth of the superposed index modulation with the number of stored gratings $M$, and the resulting per-grating diffraction efficiency under a finite per-cell dynamic range, compared with photorefractive recording. |
| [`manuscript-figures-5and6.ipynb`](manuscript-figures-5and6.ipynb) | Fig. 4-6 | Experimental demonstration of our proposed recording device |


Both run in Google Colab with no setup beyond the default environment. Each notebook contains the explanatory text, the code, and the figure; open either in Colab with the badge at the top of the file or run it locally with `numpy` and `matplotlib`.

### `DAMShift.ipynb` (Fig. 2)

The one-dimensional hidden layer is a line of spherical-reference foci, one per memory neuron, and the stored associations are separated by volume shift selectivity. The device physics follows G. Barbastathis, M. Levene, and D. Psaltis, "Shift multiplexing with spherical reference waves," *Appl. Opt.* **35**, 2403 (1996). Panel (a) evaluates the shift selectivity $\delta(L)$ and the number of resolvable memory neurons $C = s/(p\delta)$ as the hologram thickness $L$ runs from 30 µm to 10 mm, showing that $C$ saturates because the reference focus must recede as $L$ grows. Panel (b) evaluates interpage crosstalk as a function of page angular extent and converts it to recall fidelity at fixed load $M = 512$: a linear read-out degrades as pages grow, while the hidden-layer softmax restores near-ideal recall at the densest packing $p = 1$.

Baseline parameters: $\lambda_0 = 0.5$ µm, $n_0 = 1.5$, NA $= 0.5$, $\theta_S = 30^\circ$, pixel pitch 10 µm, $F = 50$ mm. For these, $C \approx 37$, $330$, and $1600$ at $L = 0.1$, $1$, and $10$ mm with $p = 2$.

### `DAMMno.ipynb` (Fig. 3)

With $M$ gratings of random phase superimposed in a cell, the RMS index modulation grows as $\sqrt{M}$ and the typical peak as $\sqrt{M\log M}$. Under a finite per-cell ceiling, a cell limited by average stored energy therefore holds per-grating efficiency $\eta \propto M^{-1}$, and a hard-clipping cell holds $\eta \propto (M\log M)^{-1}$, against the $M^{-2}$ of an optimally scheduled photorefractive medium. The notebook computes the superposition statistics by Monte Carlo (200 phase draws per $M$, $M$ from 1 to 200), converts each excursion measure to an allowed per-grating amplitude, and plots the resulting efficiency. Fitted log-log slopes are printed as a check: $-1.00$ (RMS-limited), $-1.22$ (hard-clip), $-2.00$ (photorefractive).

### `manuscript-figures-5and6.ipynb` (Figs. 4-6)

Experimental demonstrations.

## Scope

Fig. 2 uses the paraxial, Born-approximation crosstalk theory of Barbastathis et al. and treats interpage crosstalk as uncorrelated noise on the hidden-layer coefficients; crosstalk from a stored key that resembles the probe is governed by the pattern statistics of the dense associative memory and is not modeled. Fig. 3 gives scaling only; the modulation ceiling is normalized, and an absolute capacity requires the per-cell voltage swing and modulation mechanism discussed in Sec. 4 of the paper. The experimental results of Sec. 4 (opposing-diode cell) are measured data and are not reproduced here.

## Citation

If you use this code, please cite the paper above. A BibTeX entry will be added on publication.

## Contact

David J. Brady, Wyant College of Optical Sciences, University of Arizona — brady.dj@gmail.com
