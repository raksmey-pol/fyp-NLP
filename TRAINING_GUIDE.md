# Model Training Guide

## 🚀 GPU-Optimized Quick Start

### Deep Learning Models (GPU Accelerated)

```bash
python src/models/deep_learning_models.py
```

**GPU Optimizations Enabled:**

- ✅ **Mixed Precision (FP16)** - 2-3x faster training
- ✅ **XLA Compilation** - Optimized GPU kernels
- ✅ **Larger Batch Size** - 128 (vs 32 on CPU) for better GPU utilization
- ✅ **CuDNN-optimized LSTM** - Hardware-accelerated operations
- ✅ **Memory Growth** - Efficient VRAM usage

**Expected Speed (GTX 1650):**

1. LSTM: ~3-5 minutes per model
2. BiLSTM: ~4-6 minutes per model
3. CNN-LSTM: ~3-5 minutes per model

**Total training time: ~15-20 minutes** (vs ~45-60 minutes on CPU)

---

### Traditional ML Models (Fast on CPU)

```bash
python src/models/traditional_models.py
```

**Features:**

- ✅ Progress bars for each model
- ✅ Auto-handles sparse matrices
- ✅ SVM uses 5000 samples for speed
- ✅ Saves models automatically

**Training time: ~5-10 minutes total**

---

## GPU Configuration

Edit `src/models/deep_learning_models.py` (lines 15-16):

```python
USE_GPU = True          # Enable/disable GPU
OPTIMIZE_GPU = True     # Enable mixed precision & XLA
```

**Performance Modes:**
| Mode | Speed | Memory | Use When |
|------|-------|--------|----------|
| `USE_GPU=True, OPTIMIZE_GPU=True` | 🚀 Fastest | 2-3GB VRAM | GTX 1650 (recommended) |
| `USE_GPU=True, OPTIMIZE_GPU=False` | ⚡ Fast | 1-2GB VRAM | Older GPUs |
| `USE_GPU=False` | 🐌 Slow | CPU RAM | No GPU / troubleshooting |

---

## Troubleshooting

### Out of GPU Memory

**Error:** `ResourceExhaustedError: OOM when allocating tensor`

**Solutions:**

1. Reduce batch size in code (line 435):
   ```python
   batch_size = 64  # Change from 128 to 64
   ```
2. Or disable optimizations:
   ```python
   OPTIMIZE_GPU = False
   ```

### Training Hangs

**Symptom:** First epoch takes >5 minutes

**Solutions:**

1. Wait - first epoch compiles GPU kernels (1-2 min normal)
2. If still stuck, set `USE_GPU = False`

### "CUDA out of memory" or Similar

Kill other GPU processes:

```bash
nvidia-smi  # Check GPU usage
# Kill other processes using GPU
```

---

## Current Status

✅ **Phase 4 Complete!**

- Traditional ML models code ready
- Deep Learning models code ready
- Progress bars added
- GPU support with CPU fallback

## Next: Phase 5

After training completes, move to **Phase 5: Model Evaluation**

- Detailed performance analysis
- Confusion matrices
- ROC curves
- Feature importance
- Model comparison

---

## Troubleshooting

### "Shape () error"

✅ Fixed - sparse matrices now handled correctly

### "GPU hangs/freezes"

✅ Set `USE_GPU = False` in deep_learning_models.py

### "Out of memory"

✅ Batch size automatically reduced for GPU

### "SVM too slow"

✅ Using 5000 sample subset (edit `svm_sample_size` in script)
