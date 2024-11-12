using StaticArrays
function get_low_ringrank_data(R,J)
    @assert length(J) == length(R) "length of R and J is not same"
    D = length(J)
    G = Vector{Array{Float64,3}}(undef,D)
    for i = 1:D
        if i != D
            G[i] = rand(R[i],J[i],R[i+1])
        else
            G[D] = rand(R[D],J[D],R[1])
        end
    end
    T = reconst(G)
    return T
end

"""
function reconst(G)
    d = length(G)
    I = zeros(Int,d)
    for n = 1:d
        _, In, _ = size(G[n])
        I[n] = In
    end
    reconst_T = zeros(I...)

    for idx in CartesianIndices((tuple(I...)))
        reconst_T[idx] = tr(prod([G[n][:,idx[n],:] for n=1:d]))
    end
    return reconst_T
end
"""

struct MatrixArray end
matrixarray(S) = [SMatrix{size(S[:, i, :])...}(S[:, i, :]) for i in axes(S, 2)]
reconst(G) = reconst(MatrixArray(), matrixarray.(G))
reconst(::MatrixArray, G) = _reconst(G...)

function _reconst(Gs...)
    h = length(Gs) ÷ 2
    _reconst(reduce(eachprod, Gs[begin:h]), reduce(eachprod, Gs[h+1:end]))
end
_reconst(G1, G2) = trprod.(G1, expanddim(G2, G1))

trprod(A, B) = dot(vec(A'), vec(B))
eachprod(A, B) = A .* expanddim(B, A)
expanddim(B, A) = reshape(B, (ntuple(_ -> 1, ndims(A))..., size(B)...))

