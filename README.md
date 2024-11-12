
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

![ig_fig](https://github.com/user-attachments/assets/71e06933-25ad-4916-bf2d-0a22fbc20282)



# Tutorial

## (θ,η)-representation
Let's convert a 3x3x3 non-negative normalized tensor P into its θ and η representation. θ and η are multi-dimensional arrays whose size is the same as P. 

```julia
using LinearAlgebra
include("get_params.jl");

P = normalize(rand(10,10,10),1);
eta = get_eta_from_tensor(P);
theta = get_theta_from_tensor(P);
```
θ and η are often called natural parameters and expectation parameters, respectively. The transformation X ⇔ θ ⇔ η has one-to-one correspondence, and then it does not lose any information. Although each element in `P` needs to be non-negative, the parameter `θ` can be any real value. However, $\theta_{111}$ has a role of normalization and is specified by all other $\theta$ values. If $i<l, j<m, k<n$ then, $\eta_{ijk} < \eta_{lmn}$ need to be satsifeid. 

We can always recover the original tensor X from θ or η representation. 

```julia
P_from_eta   = get_tensor_from_theta(theta);
P_from_theta = get_tensor_from_eta(eta);
```

Although each element in `P` needs to be non-negative, the parameter `θ` can be any real value except for $\theta_{1111}$.  
In the following, we assume all tensors normalized (i.e. $\sum_{ijkl} P_{ijkl}=1$) and non-negative.

![ig_convert](https://github.com/user-attachments/assets/0ff6906b-0946-4d7c-9783-ce636cdde907)

Please refer to Equations (6), (7), (8), and (9) in [this paper](http://proceedings.mlr.press/v70/sugiyama17a/sugiyama17a.pdf) for the mathematical formula for transformation among X, θ, and η. 

## Many-body approximation

Many-body approximation reduces high-order interaction among tensor modes. Let us consider one-body, two-body, and three-body approximations of a given fourth-order tensor P. The $n$-body approximation of the tensor $P$ is represented as $P^{\leq n}$.

$$
\begin{align}
P_{ijkl} \simeq P_{ijkl}^{\leq 1} &= p_i^{(1)}p_j^{(2)}p_k^{(3)}p_l^{(4)} \\
P_{ijkl} \simeq P_{ijkl}^{\leq 2} &= X_{ij}^{(12)}X_{ik}^{(13)}X_{il}^{(14)}X_{jk}^{(23)}X_{kl}^{(34)} \\
P_{ijkl} \simeq P_{ijkl}^{\leq 3} &= X_{ij}Y_{ik}Z_{il}U_{jk}V_{jl}W_{kl}
\end{align}
$$

where the symbol $\simeq$ means approximation in terms of the KL divergence. Each factorization can be described by a graph called interaction representation. Please refer to the original paper to see the relationship between Interaction representation  and tensor networks. 

The following commands perform $n$-body approximation of the given normalized random non-negative tensor `P`.
```Julia
P = normalize(rand(10,10,10,10),1);
n = 3
X_nbody, theta_nbody, eta_nbody = manybody_app(X_nbody, n, verbose=true);
```

We obtain the projection destination from `P` onto the $n$-body manifold $\mathcal{B}_n$, which is a set of tensors that can be described in Equation. 
in the tensor representation, θ-representation, and η-representation. We note that `X_nbody` is globally optimal tensor that minizis the objective function

$$
\begin{align}
P = \arg\min_{Q \in \mathcal{B}} D_{KL}(P,Q)
\end{align}
$$

We belive a great contribution comparing to the tradional low-rank approximation.

If we include the hidden variables (mode) in the low-body tensor, the model will be a low-rank tensor, which forms non-convex optimisation problems, as shown in [this paper](https://arxiv.org/abs/2405.18220). 

We can customize the interaction using the list of binary vector `intracts`. In the following example, 
the fast binary vector `[1,1,1,1]` means all one-body interactions are activated. `1` means activated, `0` means deactivated. Then, the second binary vector `[1,0,1,0,0,0]` means two-body interactions (i,j) and (i,k) are activated. Each value corresponds to two body interactions `[(i,j), (i,k), (i,l), (j,k), (j,l), (k,l)]`. The third 

```julia
# define intraction with all one-body interactions and
# two-body interactions of (1,2) and (1,4) and
# three-body interactions of (1,2,3).
intracts = [ [1,1,1,1],[1,0,1,0,0,0],[1,0,0,0],[0] ];
P, theta, eta = manybody_app(T, intracts)
```

### Example for COIL Dataset

![ig_coil](https://github.com/user-attachments/assets/a097a7d7-2965-452c-ab84-da99e938f28d)


### Low-body tensor completion

## Legendre decomposition

Legendre decomposition is a generalization of many-body approximation. The binary tensor specifies which θ is to be fixed at 0. The size of this binary tensor is equal to the input tensor.

Links to 
- [Python implementation by R. Kojima](https://github.com/kojima-r/pyLegendreDecomposition)
- [C++ implementation by M. Sugiyama](https://github.com/mahito-sugiyama/Legendre-decomposition)
- [Python implementation by Y. Kawakami](https://github.com/Yhkwkm/legendre-decomposition-python)

## Legendre Tucker-rank decomposition

## Tensor balancing

Links to
- [C++ implementation by M. Sugiyama](https://github.com/mahito-sugiyama/newton-balancing)
- [Julia implementation](https://github.com/k-kitai/TensorBalancing.jl) 

# Further readings

A lot of work based on are devloping based on information

- How to choose interaction automatically? by J. Enouen [[arXiv]](https://arxiv.org/pdf/2410.11964) 
- Blind Source Separation via Legendre Transformation, by S. Luo [[Paper]](https://proceedings.mlr.press/v161/luo21a.html) [[Code]](https://github.com/sjmluo/IGLLM?utm_source=catalyzex.com) [[Slide]](https://github.com/sjmluo/IGLLM/blob/master/IGBSS_NeurIPS2020_Poster.pdf)
- Relationship between many-body approximation and low-rank approximation by K. Ghalamkari [[arxiv]](https://arxiv.org/abs/2405.18220)

# Baselines

This repository also provides the following tensor methods. 
#### Factorizations
- [CPAPR](https://epubs.siam.org/doi/abs/10.1137/110859063?casa_token=SSZGjzSrFL8AAAAA:OqRbePMSM1sTo6pV8vIsF4UhuKfu-zNfRDH7dIo8NTE8HQtlHaiYqqqcbpsxbe1VYxRbhNTTCbM): CP Alternating Poisson Regression 
- [NNTF](https://link.springer.com/chapter/10.1007/978-3-030-41032-2_17)  : Non-negative Tensor Train Factorization
- [NTD](https://ieeexplore.ieee.org/abstract/document/4270403): Non-negative Tucker decomposition
- [NTR](https://link.springer.com/article/10.1007/s11431-020-1820-x): Non-negative Tensor Ring Decomposition
- [TR](https://arxiv.org/abs/1606.05535): Tensor Ring Decomposition based on SVD
- [lraSNTD](https://ieeexplore.ieee.org/document/6166354): Sequential Nonnegative Tucker Decomposition

#### Tensor completions
- [PTRCRW](https://ieeexplore.ieee.org/document/9158539/): Low tensor-ring rank completion by parallel matrix factorization
- [SiLRTC](https://ieeexplore.ieee.org/document/6138863): Simple Low Rank Tensor Completion
- [HaLRTC](https://ieeexplore.ieee.org/document/6138863): High Accuracy Low Rank Tensor Completion
- [SiLRTCTT](https://ieeexplore.ieee.org/abstract/document/7859390): Simple Low Rank Tensor Completion with Tensor Train
- [TMacTT](https://ieeexplore.ieee.org/abstract/document/7859390): Tensor completion by parallel matrix factorization via tensor train

# Links to other packages

Many-body approximation
- [Python implementation by R. Kojima](https://github.com/kojima-r/pyLegendreDecomposition)

Legendre decomposition
- [Python implementation by R. Kojima](https://github.com/kojima-r/pyLegendreDecomposition)
- [C++ implementation by M. Sugiyama](https://github.com/mahito-sugiyama/Legendre-decomposition)
- [Python implementation by Y. Kawakami](https://github.com/Yhkwkm/legendre-decomposition-python)

Tensor balancing
- [C++ implementation by M. Sugiyama](https://github.com/mahito-sugiyama/newton-balancing)
- [Julia implementation](https://github.com/k-kitai/TensorBalancing.jl) 


# Citation
If you use this source code in a scientific publication, please consider citing the following papers:
