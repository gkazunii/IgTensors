
# What it is?

This is a julia package for non-negative tensor methods based on information goemetry, which supports following papers:
- Many-body approximation for non-negative tensors [[Paper]](https://neurips.cc/virtual/2023/poster/72780)
- Low-body tensor completion [[Paper]](https://neurips.cc/virtual/2023/poster/72780)
- Legendre decomposition [[Paper]](https://papers.nips.cc/paper_files/paper/2018/hash/56a3107cad6611c8337ee36d178ca129-Abstract.html)
- Tensor balancing [[Paper]](https://papers.nips.cc/paper_files/paper/2018/hash/56a3107cad6611c8337ee36d178ca129-Abstract.html)

Our code works on Julia 1.8. 

# Why information geometry?


Normalized nonnegative tensors have a natural correspondence with discrete probability distributions where the indices are discrete random variables and the index set is the sample space. Traditionally, tensor learning has been performed by projecting given tensor onto a model space (e.g., a set of low-rank tensors) in the coordinate system formed by each element of the tensor (P_111,P_211,...,P_IJK). However, this projection is generally a nonconvex optimization problem.

Information geometry allows probability distributions to be represented in a convenient dual-flat coordinate system, the θ and η coordinate systems. These coordinate systems make it simple to discuss how the model space should be defined to formulate learning as a convex optimization problem. 


# Tutorial

### (θ,η)-representation
Let's convert a 3x3x3 non-negative normalized tensor X into its θ and η representation.

The correspondense X ⇔ θ ⇔ η is one-to-one and the transfomation lose any infromation. We can always recover the original tensor X from θ or η reprsentation. 

### Many-body approximation

If we include the hidden variables (mode) in the low-body tensor, the model will be low-rank tensor, which form non-convex optimziation problems, as shown in [this paper](https://arxiv.org/abs/2405.18220). 

### Legendre decomposition

Legendre decomposition is a geneliazation of many-body approximation. 

### Legendre Tucker-rank decomposition

### Tensor balancing

# Baselines

This repository also provides following tensor methods. 
#### Factorizations
- CPAPR
- NNTF: Non-negative Tensor Train Factorization
- NTD: Non-negative Tucker decomposition
- NTR: Non-negative Tensor Ring Decomposition
- TR: Tensor Ring Decomposition based on SVD
- [lraSNTD](https://ieeexplore.ieee.org/document/6166354): Sequential Nonnegative Tucker Decomposition

#### Tensor completions
- [PTRCRW](https://ieeexplore.ieee.org/document/9158539/): Low tensor-ring rank completion by parallel matrix factorization
- [SiLRTC](https://ieeexplore.ieee.org/document/6138863): Simple Low Rank Tensor Completion
- [HaLRTC](https://ieeexplore.ieee.org/document/6138863): High Accuracy Low Rank Tensor Completion
- [SiLRTCTT](https://ieeexplore.ieee.org/abstract/document/7859390): Simple Low Rank Tensor Completion with Tensor Train
- [TMacTT]((https://ieeexplore.ieee.org/abstract/document/7859390)): Tensor completion by parallel matrix factorization via tensor train
- 

# Citation
If you use this source-code in a scientific publication, please consider cite following papers:
