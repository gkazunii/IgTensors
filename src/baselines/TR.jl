using LinearAlgebra
using Glob
using Plots
using Images
using TensorToolbox
using Statistics
using StaticArrays

function get_n_params_ring(R,J)
    n_params = 0
    D = length(J)
    for i = 1:D
        if i != D
            n_params += R[i]*J[i]*R[i+1]
        else
            n_params += R[D]*J[D]*R[1]
        end
    end
    return n_params
end

# ３次元配列を行列の配列に変換する
"Just specify that [an element of] the argument is an array of matrices"
struct MatrixArray end

"""
    R = matrixarray(S::AbstractArray{T,3})

`R[k][i, j] = S[i, k, j]`
"""
matrixarray(S) = [SMatrix{size(S[:, i, :])...}(S[:, i, :]) for i in axes(S, 2)]
#matrixarray(S) = [S[:, i, :] for i in axes(S, 2)] # without StaticArrays


# 行列積の最終段を省き、トレースを直接求める
reconst(G) = reconst(MatrixArray(), matrixarray.(G))
reconst(::MatrixArray, G) = _reconst(G...)

# Gs を真ん中で分けないと遅くなる
function _reconst(Gs...)
    h = length(Gs) ÷ 2
    _reconst(reduce(eachprod, Gs[begin:h]), reduce(eachprod, Gs[h+1:end]))
end

#_reconst(G1, G2, Gs...) = _reconst(eachprod(G1, G2), Gs...) # 遅い
_reconst(G1, G2) = trprod.(G1, expanddim(G2, G1))

"""
    trprod(A, B)

Returns `tr(A * B)`
"""
trprod(A, B) = dot(vec(A'), vec(B))

"""
    C = eachprod(A, B)

`C[i, j, k] = A[i, j] * B[k]`

`A, B, C :: Array{Matrix}`
"""
eachprod(A, B) = A .* expanddim(B, A)


"""
    Bx = expanddim(B, A)

`Bx = reshape(B, (1, 1, 1, m, n))` where `ndims(A) == 3`, `size(B) == (m, n)`
"""
expanddim(B, A) = reshape(B, (ntuple(_ -> 1, ndims(A))..., size(B)...))

"""
Provided by http://www.nct9.ne.jp/m_hiroi/light/juliaa01.html
"""
function divisor(n)
    # pq の約数を求める
    divisor_sub(p, q) = [p ^ i for i = 0 : q]

    xs = factorization(n)
    ys = divisor_sub(xs[1]...)
    for p = xs[2 : end]
        ys = [x * y for x = divisor_sub(p...) for y = ys]
    end
    sort(ys)
end

function factorSub(n, m)
    c = zero(n)
    while n % m == 0
        c += 1
        n = div(n, m)
    end
    c, n
end

function factorization(n)
    x::typeof(n) = 2
    xs = Pair{typeof(n), typeof(n)}[]
    c, m = factorSub(n, x)
    if c > 0
        push!(xs, x => c)
    end
    x = 3
    while x * x <= m
        c, m = factorSub(m, x)
        if c > 0
            push!(xs, x => c)
        end
        x += 2
    end
    if m > 1
        push!(xs, m => one(n))
    end
    xs
end

function lowrank_app(M,r::Int)
    U,Σ,V = svd(M)
    Ur = U[:,1:r]
    Σr = Σ[1:r]
    Vr = V[:,1:r]
    # reconstraction is Ur*diagm(Σr)*Vr'
    return Ur,diagm(Σr),Vr
end

function lowrank_app_th(M,δ)
    m = rank(M)
    U,Σ,V = svd(M)
    sorted_Σ = sort(Σ)
    err = cumsum(sorted_Σ[begin:end-1].^2)
    reverse!(err)

    r = m
    for l = 1:m-1
        if err[l] < δ
            r = l
            break
        end
    end

    Ur = U[:,1:r]
    Σr = Σ[1:r]
    Vr = V[:,1:r]
    # reconstraction is Ur*diagm(Σr)*Vr'
    return Ur,diagm(Σr),Vr,r
end

"""
Tensor Ring Decomposition based on SVD
proposed by [Q Zhao (2016)](https://arxiv.org/abs/1606.05535)

[Matlab code](https://github.com/oscarmickelin/tensor-ring-decomposition)
is also available,

# Aruguments
- 'T' : input tensor
- 'r' : target rank, which should be vector

example:
    T = randn(20,20,30)
    r = (5,3,10)
    ring_cores = TR(T,r)
    size.(ring_cores)
    # 3-element Vector{Tuple{Int64, Int64, Int64}}:
    # (5, 20, 3)
    # (3, 20, 10)
    # (10, 30, 5)

    # We can obtain low-ring rank tensor by 'reconst'
    Tr = reconst(ring_cores)
"""
function TR(T,r::Vector)
    n = size(T)
    d = length(r)
    @assert r[1]*r[2] <= n[1] "r1*r2 should be smaller than n[1]"
    @assert d == ndims(T) "the length of ranks should be same as the dims of tensor"
    r0 = r[1]*r[2]
    T1 = reshape(T,(n[1],prod(n[2:end])))
    U,Σ,V = lowrank_app(T1,r0)
    G = []
    G1 = permutedims(reshape(U,(n[1],r[1],r[2])),[2,1,3])
    C  = permutedims(reshape(Σ*V',(r[1],r[2],prod(n[2:d]))),[2,3,1])
    push!(G,G1)
    for k=2:d-1
        C = reshape(C,(r[k]*n[k],prod(n[k+1:end])*r[1]))
        U,Σ,V = lowrank_app(C,r[k+1])
        Gk = reshape(U,(r[k],n[k],r[k+1]))
        push!(G,Gk)
        C = reshape(Σ*V',(r[k+1],:,r[1]))
    end
    Gd = reshape(C,(r[d],n[d],r[1]))
    push!(G,Gd)
    return G
end

"""
Tensor ring decomposition with prescribed error value ε.
The reconstraction error becomes smaller than ε.
See more details in the [paper](https://arxiv.org/abs/1606.05535)

# Aruguments
- 'T' : input tensor
- 'ε' : prescribed error, which should be float64

example:
    Tr = reconst(ring_cores)
    T = rand(10,20,7)
    ε = 0.3
    G,r = TR(T,ε);
    Tre = reconst(G)
    @show norm(T - Tre)/norm(T) < ε
    # true

    Gnew = TR(T,r);
    for l = 1:length(r)
        @show Gnew[l] ≈ G[l]
        # true
    end
"""
function TR(T,ε::Float64)
    n = size(T)
    d = ndims(T)
    r = zeros(Int,d)
    G = []

    δ1 = sqrt(2/d)*ε*norm(T)
    δ = δ1/sqrt(2)
    T1 = reshape(T,(n[1],prod(n[2:end])))
    U,Σ,V, r0 = lowrank_app_th(T1,δ1)
    factors = divisor(r0)
    r[1] = factors[floor(Int,length(factors)/2)]
    r[2] = Int( r0 / r[1] )

    G1 = permutedims(reshape(U,(n[1],r[1],r[2])),[2,1,3])
    C  = permutedims(reshape(Σ*V',(r[1],r[2],prod(n[2:d]))),[2,3,1])
    push!(G,G1)
    for k=2:d-1
        C = reshape(C,(r[k]*n[k],prod(n[k+1:end])*r[1]))
        U,Σ,V, rnew = lowrank_app_th(C,δ)
        r[k+1] = rnew
        Gk = reshape(U,(r[k],n[k],r[k+1]))
        push!(G,Gk)
        C = reshape(Σ*V',(r[k+1],:,r[1]))
    end
    Gd = reshape(C,(:,n[d],r[1]))
    r[d] = size(Gd)[1]
    push!(G,Gd)
    return G, r
end
