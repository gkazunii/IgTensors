using TensorToolbox
using LinearAlgebra
using Arpack
using TensorDecompositions
using TensorToolbox

function CP_als(T, reqrank)
    @assert length(Set(reqrank)) == 1 "Core Tensor should be cubic"
    reqrank = reqrank[1]
    F = cp_als(T, reqrank)
    D = ndims(T)

    F1 = F.lambda[1] * ttt(F.fmat[1][:,1], F.fmat[2][:,1])
    for d = 3:D
        F1 = ttt(F1, F.fmat[d][:,1])
    end
    for r = 2:reqrank
        Fr = F.lambda[r] * ttt(F.fmat[1][:,r], F.fmat[2][:,r])
        for d = 3:D
            Fr = ttt(Fr, F.fmat[d][:,r])
        end
        F1 .= F1 .+ Fr
    end
    return F1
end

function CP(T, reqrank ; method="ALS")
    # method is "ALS" or "SGSD"
    # ALS:
    # epubs.siam.org/doi/pdf/10.1137/S089547980139786X
    # SGSD:
    # epubs.siam.org/doi/pdf/10.1137/07070111X
    #
    # input is any 3-way tensor for SGSD
    # input is any tensor for ALS
    # cost function is L2
    @assert length(Set(reqrank)) == 1 "Core Tensor should be cubic"
    reqrank = reqrank[1]

    N = ndims(T)
    input_tensor_shape = size(T)
    init_pos = []
    for i=1:N
        push!(init_pos, randn(input_tensor_shape[i], reqrank))
    end

    # decompose
    F = candecomp(T, reqrank, Tuple(init_pos), verbose=false, method=Symbol(method))

    G = zeros( fill(reqrank, N)... )
    # get core tensor
    for i = 1:reqrank
        idx = fill(i, N)
        G[idx...] = F.lambdas[i]
    end

    # reconstract
    U = F.factors
    X = ttm(G, U[1], 1)
    for n = 2:N
        X = ttm(X, U[n], n)
    end
    return X
end

function NNCP(T, reqrank)
    # archive.ymsc.tsinghua.edu.cn/pacm_download/265/8687-BCD_for_multiconvex_Opt.pdf
    # CP deompositon by block-coordinate update method
    # input is non-negative tensor
    # cost function is L2
    @assert length(Set(reqrank)) == 1 "Core Tensor should be cubic"
    #G = zeros(reqrank...)
    reqrank = reqrank[1]
    N = ndims(T)
    input_tensor_shape = size(T)

    # decompose
    F = nncp(T, reqrank,compute_error=true)

    G = zeros( fill(reqrank, N)... )
    # get core tensor
    for i = 1:reqrank
        idx = fill(i, N)
        G[idx...] = F.lambdas[i]
    end

    # reconstract
    U = F.factors
    X = ttm(G, U[1], 1)
    for n = 2:N
        X = ttm(X, U[n], n)
    end
    return X
end
