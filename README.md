# Real Time Efficient TAM

### Getting Started
- python >= 3.10

### 1. Installation
Create a virtual environment with **Python 3.10** and activate it
Then, install RealtimeEfficientTam; 
```bash
git clone https://github.com/AllenNeuralDynamics/RealtimeEfficientTAM.git
cd RealtimeEfficientTAM
pip install -e .
```

### 2. Download Checkpoints
```bash
cd checkpoints
./download_checkpoints.sh
```
or EfficientTAM checkpoints are available at [the Hugging Face Space](https://huggingface.co/yunyangx/efficient-track-anything/tree/main).

### 3. Run Example
```bash
python -m efficient_track_anything.demo
```

### 4. Usage
Building a predictor:
```bash
from efficient_track_anything.realtime_tam import build_predictor
# Build the predictor, which handles model loading and device setup
predictor = build_predictor()
```

Track:
```bash
# Start a new track
from efficient_track_anything.realtime_tam import start

# Load your initial frame (NumPy array/OpenCV image)
initial_frame = load_your_frame(...) 

# Load the first frame into the predictor's state
predictor.predictor.load_first_frame(initial_frame)

# Define user prompts 
points = np.array([[x1, y1], [x2, y2]], dtype=np.float32) 
labels = np.array([1, 1], dtype=np.int32) # 1 for foreground, 0 for background

# Run the initial detection
# Returns: (None, mask_logits)
_, out_mask_logits = start(predictor, points=points, labels=labels)
```

### License
Efficient track anything checkpoints and codebase are licensed under Apache 2.0.
Implementation of real-time EfficientTAM[📕Project]

### 🙏 Acknowledgements 
We gratefully acknowledge the contributions of the developers at Meta and GitHub for making these innovative projects available to the open-source community.
This work builds upon the following projects:
- [SAM2](https://github.com/facebookresearch/segment-anything-2)  
- [EfficientTAM](https://github.com/facebookresearch/EfficientTAM)  
- [segment-anything-2-real-time](https://github.com/Gy920/segment-anything-2-real-time)


### 📖 Citation

If you use this repository in your research or applications, please cite **EfficientTAM**:

```bibtex
@article{xiong2024efficienttam,
  title={Efficient Track Anything},
  author={Yunyang Xiong, Chong Zhou, Xiaoyu Xiang, Lemeng Wu, Chenchen Zhu, Zechun Liu, Saksham Suri, 
          Balakrishnan Varadarajan, Ramya Akula, Forrest Iandola, Raghuraman Krishnamoorthi, 
          Bilge Soran, Vikas Chandra},
  journal={arXiv preprint arXiv:2411.18933},
  year={2024}
}
```

---

### Code Quality Check

The following are tools used to ensure code quality in this project. 

- Unit Testing

```bash
uv run pytest tests
```

- Linting

```bash
uv run ruff check
```

- Type Check

```bash
uv run mypy src/mypackage
```

## Documentation
To generate the rst files source files for documentation, run
```bash
sphinx-apidoc -o docs/source/ src
```
Then to create the documentation HTML files, run
```bash
sphinx-build -b html docs/source/ docs/build/html
```
