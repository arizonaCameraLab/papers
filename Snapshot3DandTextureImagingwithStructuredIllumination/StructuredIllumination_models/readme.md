````markdown
# Projector–Camera RGB Reconstruction (Structured Illumination)

This repository implements a projector–camera structured-illumination pipeline
for joint **RGB reflectance** and **surface geometry** reconstruction. The system
combines a calibrated projector–camera forward model with a **differentiable
projector–camera model (DPCM)** and a neural reconstruction framework.

**Entry point:** `Proj_cam_RGB.ipynb`

---

## Directory Structure

```text
.
├── Calibration/                  # Projector–camera calibration data and scripts
├── funcs/                        # Core modules
│   ├── DIP_helper_fun.py         # Deep Image Prior helper functions
│   ├── Unet_skip.py              # U-Net with skip connections
│   ├── proj_cam_model.py         # Differentiable projector–camera model (DPCM)
│   ├── NetParallelTrain.py       # Parallel network training utilities
│   └── utils.py                  # Utility functions
├── NN_train_results/             # Network training outputs and checkpoints
├── Patterns/                     # Projected illumination patterns
├── Real_data/                    # Real captured measurements
├── Reference_data/               # Reference / ground-truth data
├── blender_model_compare.ipynb   # Blender vs. real vs. DPCM comparison
├── Data_comp_ref.ipynb           # Data vs. reference comparison notebook
├── Proj_cam_RGB.ipynb            # Main notebook (entry point)
└── calibrated_pattern_coeff.npy  # Calibrated projector pattern coefficients
````

---

## Usage Workflow

Follow the steps below to run the complete projector–camera reconstruction
pipeline.

---

### 1. Import Calibration Parameters

Obtain the projector–camera calibration JSON file from the external repository:

```
projector-camera-control-system/
```

Copy the calibration JSON file into:

```
Calibration/
```

---

### 2. Generate Blender-Rendered Calibration Data

Run the external simulation repository:

```
Blender-Projector–Camera-Simulation/
```

Using the calibration JSON file from Step 1, generate Blender-rendered **TIFF**
images of a planar white board.

After rendering, copy the generated TIFF files into:

```
Calibration/
```

---

### 3. Validate Calibration Consistency

Run the notebook:

```
blender_model_compare.ipynb
```

This notebook compares the following three results:

1. **Blender-rendered** planar white board
2. **Real captured** planar white board measurement
3. **DPCM-rendered** (Differentiable Projector–Camera Model) result

All three results should be **numerically and visually consistent**.

**Important notes:**

* Blender does **not** support rendering cases where the focal lengths differ
  ((f_x != f_y)); such cases must be tested separately.
* The DPCM supports (f_x != f_y) and can therefore better match real
  measurements.
* Calibration is considered valid when the **real measurement and DPCM output
  closely agree**.

Once this condition is met, proceed to the reconstruction stage.

---

### 4. Run RGB and Geometry Reconstruction

Open and run the main notebook:

```
Proj_cam_RGB.ipynb
```

In the **first cell**, select the object to be reconstructed:

```python
obj = "box"   # options: "box", "cup", "buzz"
```

---

### 5. Retrieve Results

All reconstruction outputs, including intermediate results and trained network
checkpoints, are automatically saved to:

```
NN_train_results/
```

---

## Notes

* Accurate projector–camera calibration is critical for successful
  reconstruction.
* Validation using a planar white board is strongly recommended before running
  reconstructions on complex objects.
* The DPCM provides a physically consistent forward model and enables accurate
  reconstruction even when projector intrinsics are anisotropic
  ((f_x != f_y)).

---

## How to Cite

If you use this code or its associated ideas in academic work, please cite the
corresponding publication or thesis:

```bibtex
@phdthesis{Dong2026DPCM,
  author       = {Zhipeng Dong},
  title        = Snapshot 3D and Texture Imaging with Structured Illumination},
  school       = {University of Arizona},
  year         = {2026}
}
```

If a journal or conference paper is preferred, please replace the above entry
with the appropriate citation.

---

## Author

**Zhipeng Dong**
Ph.D. Candidate
University of Arizona

For questions or issues related to this repository, please contact the author.

```
```
