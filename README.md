# Real Time Efficient TAM

### Getting Started
- python >= 3.10

### 1. Installation
```bash
git clone https://github.com/AllenNeuralDynamics/RealtimeEfficientTAM.git
cd RealtimeEfficientTAM
conda create -n rttam python=3.10
conda activate rttam
pip install -e .
```
### 2. Download Checkpoints
```bash
cd checkpoints
./download_checkpoints.sh
```
or EfficientTAM checkpoints are available at the Hugging Face Space.

### 3. Run 
TBD

---
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
