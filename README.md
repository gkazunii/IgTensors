
# What it is?

This is a julia package for non-negative tensor methods based on information goemetry, which supports following papers:
- Many-body approximation for non-negative tensors [[Paper]](https://neurips.cc/virtual/2023/poster/72780) [[Poster]](https://nips.cc/media/PosterPDFs/NeurIPS%202023/72780.png?t=1698249769.865172) [[Slide]]()
- Low-body tensor completion [[Paper]](https://neurips.cc/virtual/2023/poster/72780)
- Legendre decomposition [[Paper]](https://papers.nips.cc/paper_files/paper/2018/hash/56a3107cad6611c8337ee36d178ca129-Abstract.html) [[Poster]](https://mahito.nii.ac.jp/2ba8ffb6f7afa5b021d6c57555b16f04/Sugiyama_NeurIPS2018_poster.pdf) [[Slide]](https://mahito.nii.ac.jp/60230e98c12af12f4dacb0dab21e5ec9/Sugiyama_NeurIPS2018_slide.pdf)
- Tensor balancing [[Paper]](https://papers.nips.cc/paper_files/paper/2018/hash/56a3107cad6611c8337ee36d178ca129-Abstract.html) [[Poster]](https://mahito.nii.ac.jp/c8b7b54d22b622cc389d16dba5a96543/Sugiyama_ICML2017_poster.pdf) [[Slide]](https://mahito.nii.ac.jp/3917bf4c2ee058ed7e8816a86d8c1047/Sugiyama_ICML2017_slide.pdf)

Our code works on Julia 1.8. 

# Why information geometry?


Normalized nonnegative tensors have a natural correspondence with discrete probability distributions where the indices are discrete random variables and the index set is the sample space. Traditionally, tensor learning has been performed by projecting given tensor onto a model space (e.g., a set of low-rank tensors) in the coordinate system formed by each element of the tensor $(P_{111}, P_{211},...,P_{IJK})$. However, this projection is generally a nonconvex optimization problem.

Information geometry allows probability distributions to be represented in a convenient dual-flat coordinate system, the θ and η coordinate systems. These coordinate systems make it simple to discuss how the model space should be defined to formulate learning as a convex optimization problem. 


# Tutorial

### (θ,η)-representation
Let's convert a 3x3x3 non-negative normalized tensor X into its θ and η representation.

The transformation X ⇔ θ ⇔ η has one-to-one correspondence, and then it does not lose any information. We can always recover the original tensor X from θ or η representation. In the following, we assume all tensors normalized (i.e. $\sum_{ijkl} P_{ijkl}=1$) and non-negative.

### Many-body approximation

Many-body approximation reduces high-order interaction among tensor modes. As an example, let us consider a 3x3x3x3 tensor P. 

$$
\begin{align}
P_{ijkl} \simeq P_{ijkl}^{\leq 1} &= a_ib_jc_kd_l \\
P_{ijkl} \simeq P_{ijkl}^{\leq 2} &= X_{ij}Y_{ik}Z_{il}U_{jk}V_{jl}W_{kl} \\
P_{ijkl} \simeq P_{ijkl}^{\leq 3} &= X_{ij}Y_{ik}Z_{il}U_{jk}V_{jl}W_{kl}
\end{align}
$$

If we include the hidden variables (mode) in the low-body tensor, the model will be low-rank tensor, which form non-convex optimziation problems, as shown in [this paper](https://arxiv.org/abs/2405.18220). 

### Legendre decomposition

Legendre decomposition is a generalization of many-body approximation. The binary tensor specifies which θ is to be fixed at 0. The size of this binary tensor is equal to the input tensor.

### Legendre Tucker-rank decomposition

### Tensor balancing

# Baselines

This repository also provides the following tensor methods. 
#### Factorizations
- CPAPR: CP Alternating Poisson Regression 
- [NNTF](https://link.springer.com/chapter/10.1007/978-3-030-41032-2_17): Non-negative Tensor Train Factorization
- [NTD](https://ieeexplore.ieee.org/abstract/document/4270403): Non-negative Tucker decomposition
- [NTR](https://link.springer.com/article/10.1007/s11431-020-1820-x): Non-negative Tensor Ring Decomposition
- [TR](https://arxiv.org/abs/1606.05535): Tensor Ring Decomposition based on SVD
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
