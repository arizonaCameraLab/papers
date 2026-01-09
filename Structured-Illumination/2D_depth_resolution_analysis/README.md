## Overview

The tool models the 3D measurement volume in a 2D $xz$-plane to evaluate how extrinsic device parameters—specifically the camera and projector positions and their respective rotation angles ($\theta_{\rm cam}$, $\theta_{\rm proj}$)—influence the fundamental limits of depth discrimination.



## Mathematical Methodology

The simulation follows the rigorous triangulation model described in the associated research:

### 1. Ray Parametrization
Each pixel boundary is treated as a discrete interval, defining divergent rays originating from the optical centers ($C_{\rm cam}$, $C_{\rm proj}$). The rays are represented by parametric line equations:

$$l(\tau) = C + \tau d$$

where $d$ is the normalized direction vector.

### 2. Uncertainty Cell Computation
The vertices $V_{i,j}$ of the quadrangular **Uncertainty Cells** are determined by solving the linear system:

$$\begin{pmatrix} d_{\rm cam} & -d_{\rm proj} \end{pmatrix} \begin{pmatrix} \tau_{\rm cam} \\ \tau_{\rm proj} \end{pmatrix} = C_{\rm proj} - C_{\rm cam}$$

Numerical stability is ensured through a **Least-Squares** approach.



### 3. Depth Resolution Quantification
The **Depth Resolution ($\Delta Z$)** is quantified as the maximum vertical extent of the resulting intersection cell:

$$\Delta Z = \max(Z_{V_{i,j}}) - \min(Z_{V_{i,j}})$$

This value represents the spatial quantization limit inherent to the hardware configuration.

## Key Features

* **Spatial Dependency Analysis**: Evaluates how $\Delta Z$ fluctuates based on the convergence angles relative to the target surface.
* **Discrete Ray Tracing**: Renders high-fidelity ray networks to visualize the geometric sampling density.
* **Synthetic Surface Profiling**: Tests system performance against realistic range on curved surfaces.
* **Trade-off Modeling**: Analyzes the relationship between correlation patch size and spatial resolution loss.

## Configuration Guide

The system configuration is managed via the `systems` dictionary within the core script. You can adjust the parameters to simulate various hardware setups.

### 1. Hardware Extrinsics

| Parameter | Unit | Description |
| :--- | :--- | :--- |
| `focal_length` | pixel | **Effective Focal Length**. Represents the ratio between focal distance and pixel size. |
| `num_pixels` | count | **Sensor Resolution**. Defines the total discrete elements on the sensor/DMD. |
| `pixel_pitch` | mm | **Pixel Interval**. Increasing this simulates larger correlation patch sizes. |
| `theta_deg` | degree | **Convergence Angle**. Rotation of the device relative to the world $Z$-axis. |
| `dx`, `dy` | mm | **Optical Center**. Translational coordinates ($C_x, C_z$) in the $xz$-plane. |

### 2. Implementation Example
```python
systems = {
    "camera": {
        "focal_length": 4550.58,
        "num_pixels": 928,
        "theta_deg": -1.59,
        "dx": 45.97, "dy": 1656.82
    },
    "projector": {
        "focal_length": 2405.99,
        "num_pixels": 768,
        "theta_deg": -17.35,
        "dx": 440.23, "dy": 1408.60
    }
}

