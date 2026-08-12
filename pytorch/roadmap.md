# PyTorch Syllabus

**A Deep Learning Framework for Building & Training Neural Networks**

---

## → Phase 1: Tensors

**Goal:** Learn PyTorch's core data structure and how it relates to NumPy

**Topics Covered:**

- Creating tensors: from data, factory functions (`zeros`/`ones`/`rand`/`randn`/`arange`/`linspace`/`eye`)
- Tensor attributes: `shape`, `dtype`, `device`, `ndim`, `numel`
- dtype casting and device placement (cpu / cuda / mps)
- Indexing, slicing, boolean masking, fancy indexing
- Element-wise ops, broadcasting, matrix multiplication, reductions, in-place ops
- Reshaping: `view` vs `reshape`, `transpose`, `permute`, `squeeze`/`unsqueeze`, `flatten`
- The NumPy bridge: shared memory, `.clone()`

**Notebook:** `phase1.ipynb`

---

## → Phase 2: Autograd

**Goal:** Understand automatic differentiation — how PyTorch computes gradients

**Topics Covered:**

- `requires_grad`, the computation graph, `grad_fn`, leaf tensors
- `loss.backward()`, verifying gradients by hand (chain rule)
- `torch.no_grad()` for weight updates
- Why gradients accumulate, and why they must be zeroed each step
- A full manual training loop (gradient descent from scratch)
- `.detach()` vs `torch.no_grad()`, common `RuntimeError`s
- Preview: `nn.Linear` + `optim.SGD` replacing the manual version

**Notebook:** `phase2.ipynb`

---

## → Phase 3: Building Models with `torch.nn`

**Goal:** Learn `nn.Module`, the standard training loop, and batching

**Topics Covered:**

- `nn.Parameter`, writing a custom `nn.Module` subclass
- `nn.Linear`, `nn.Sequential`, activation functions
- Loss functions: `MSELoss` (regression), `CrossEntropyLoss` (classification)
- Optimizers: `SGD` vs `Adam`
- The standard training loop pattern (`zero_grad` → forward → loss → `backward` → `step`)
- `model.train()` vs `model.eval()`
- Saving/loading a `state_dict`
- `Dataset` & `DataLoader` for batching

**Assignment/Project:** Build and train an MLP classifier on MNIST — flatten images, a multi-layer `nn.Sequential`, evaluate accuracy on a genuinely held-out test set

**Notebooks:** `phase3.ipynb`, `phase3_project.ipynb`

---

## → Phase 4: Convolutional Neural Networks (CNNs)

**Goal:** Use convolution to exploit the spatial structure in images

**Topics Covered:**

- `nn.Conv2d`: `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`
- `nn.MaxPool2d`, feature maps
- Tracing tensor shapes through a conv/pool stack by hand
- Building a CNN classifier (`features` extractor + `classifier` head)

**Assignment/Project:** Train a CNN on MNIST and compare parameter count & accuracy against the Phase 3 MLP baseline

**Notebook:** `phase4.ipynb`

---

## → Phase 5: Transfer Learning & Pretrained Models

**Goal:** Learn to reuse large pretrained models instead of training from scratch

**Topics Covered:**

- `torchvision.models`: loading a pretrained architecture (ResNet) and its ImageNet weights
- Why preprocessing must match training: `weights.transforms()`, ImageNet normalization stats
- Feature extraction (freeze the backbone, train only a new head) vs fine-tuning (unfreeze some/all layers)
- Swapping the final classification layer for a new number of classes
- Differential learning rates — small `lr` for pretrained layers, larger `lr` for the new head
- When to freeze vs fine-tune, based on dataset size and similarity to ImageNet

**Assignment/Project:** Fine-tune a pretrained ResNet on a small custom image subset and compare it against training a small CNN from scratch on the *same* tiny dataset

**Notebook:** `phase5.ipynb`

---

## → Phase 6: Training Techniques That Matter

**Goal:** Learn the practical techniques that separate a model that "runs" from one that generalizes well

**Topics Covered:**

- Diagnosing overfitting vs underfitting from train/validation loss curves
- Data augmentation: `torchvision.transforms` (flips, crops, color jitter)
- Regularization: `Dropout`, weight decay (L2), `BatchNorm`
- Learning rate scheduling: `StepLR`, `ReduceLROnPlateau`, cosine annealing
- Early stopping and checkpointing the best model
- Train / validation / test split discipline

**Assignment/Project:** Take the Phase 5 model, add augmentation + an LR scheduler + early stopping, and show the validation curve improve versus the un-regularized baseline

**Notebook:** `phase6.ipynb`

---

## → Phase 7: Sequence Models (RNN / LSTM / GRU)

**Goal:** Handle sequential data where order matters (text, time series)

**Topics Covered:**

- Why a plain feed-forward network can't handle variable-length sequences
- `nn.RNN`, `nn.LSTM`, `nn.GRU`: hidden state, sequence-to-one vs sequence-to-sequence
- Padding & packing variable-length sequences (`pad_sequence`, `pack_padded_sequence`)
- `nn.Embedding` — turning tokens into learned vectors
- Vanishing gradients, and why LSTM/GRU exist

**Assignment/Project:** Build a sentiment classifier (or a simple time-series forecaster) using an LSTM

**Notebook:** `phase7.ipynb`

---

## → Phase 8: Attention & Transformers

**Goal:** Understand the architecture behind every modern LLM

**Topics Covered:**

- The limits of RNNs that motivated attention
- Self-attention from scratch: queries, keys, values, scaled dot-product attention
- Multi-head attention
- Positional encoding (attention has no built-in notion of order)
- `nn.TransformerEncoder` / `nn.TransformerEncoderLayer`
- Building a small transformer for a toy task

**Assignment/Project:** Implement scaled dot-product attention from raw tensor ops (no `nn.MultiheadAttention`), then verify it matches PyTorch's built-in layer

**Notebook:** `phase8.ipynb`

---

## → Phase 9: Practical NLP with Pretrained Transformers

**Goal:** Bridge from "understands transformers" to "can fine-tune a real language model"

**Topics Covered:**

- Tokenization: subword tokenizers (BPE/WordPiece), the Hugging Face `transformers` library
- Loading a pretrained model (e.g. a small BERT/DistilBERT) and its tokenizer
- Fine-tuning for a downstream task (text classification)
- Using a PyTorch `DataLoader` with Hugging Face `Datasets`
- Inference: pipelines vs a manual forward pass

**Assignment/Project:** Fine-tune a pretrained transformer for a text classification task and evaluate accuracy/F1 on a held-out set

**Notebook:** `phase9.ipynb`

---

## → Phase 10: Deployment & Production

**Goal:** Take a trained model from a notebook to something servable

**Topics Covered:**

- Exporting: TorchScript (`torch.jit.trace`/`script`), ONNX export
- Quantization for faster/smaller inference
- Serving a model behind a FastAPI endpoint
- Batching requests, latency vs throughput trade-offs
- Versioning models, basic monitoring for drift

**Assignment/Project:** Wrap a trained model in a FastAPI endpoint that accepts input and returns a prediction, and export it to TorchScript or ONNX

**Notebook:** `phase10.ipynb`

---

## → Optional Branch: Generative Models

**Goal:** Explore models that generate rather than classify

**Topics Covered:**

- Autoencoders: encoder/decoder, reconstruction loss
- Variational Autoencoders (VAEs): the reparameterization trick
- GANs: generator vs discriminator, adversarial training
- Diffusion models (brief) — the modern standard for image generation

**Assignment/Project:** Train a VAE or a simple GAN on MNIST and visualize the generated digits

**Notebook:** `phase-gen.ipynb`
