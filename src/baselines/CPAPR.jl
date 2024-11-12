using LinearAlgebra
using TensorDecompositions
using TensorToolbox
using InvertedIndices

function get_M(I, N, R)
    #A = []
    A = Vector{Array{Float64,2}}(undef, N)
    for i = 1 : N
        a = rand(I[i], R)
        for j = 1:R
            coin = rand()
            if coin < (1.0 / R)
                a[j] = 100*rand()
            else
                a[j] = rand()
            end
        end

        for r = 1 : R
            a[:,r] = normalize(a[:,r],1)
        end
        #push!(A, a)
        A[i] = a
    end
    lamb = rand(R)
    return lamb, A
end

function getS(k, In, R, An, Phin, kap, kaptol)
    S = zeros(In, R)
    if k == 1
        return S
    else
        for i = 1 : In
            for r = 1 : R
                if An[i,r] < kaptol && Phin[i,r] > 1.0
                    S[i, r] = kap
                else
                    S[i, r] = 0.0
                end
            end
        end
        return S
    end
end

"""
HyperParameters are same as MatLab implementation.
https://gitlab.com/tensors/tensor_toolbox/-/blob/master/cp_apr.m
"""
function CPAPR(X , R ;
        kmax = 1000, lmax = 10, tau = 1.0e-4, kap = 0.01, kaptol = 1.0e-10, epsilon = 1.0E-10, alert_convergence = false )
    I = size(X)
    N = ndims(X)

    lamb, A = get_M(I, N, R)
    #A[i] \in R^{I[i] \times R}

    Phi = []
    for n = 1 : N
        push!(Phi, rand(I[n], R))
    end

    S = undef
    for k = 1 : kmax

        isConverged = true
        for n = 1 : N
            S = getS(k, I[n], R, A[n], Phi[n], kap, kaptol)
            B = (A[n]+S) * diagm( vec(lamb) )
            PI = TensorDecompositions.khatrirao( reverse(A[Not(n)]) )'

            for l = 1:lmax
                Phi[n] = ( tenmat(X,n) ./ max.(B*PI,epsilon) ) * PI'
                if sum(abs.( min.(B, ones(size(Phi[n])) - Phi[n]) )) < tau
                    break
                end
                isConverged = false
                B .= B .* Phi[n]
            end
            lamb = ones(1, size(B)[1]) * B
            A[n] = B * diagm( vec( 1 ./ lamb )  )
        end
        if isConverged == true
            break
        end
        if k == kmax
            if alert_convergence
                println("NOT CONVERGE k=$k")
            end
        end
    end

    #reconstract
    G = zeros( fill(R, N)... )
    for i = 1:R
        idx = fill(i, N)
        G[idx...] = lamb[i]
    end

    Y = ttm(G, A[1], 1)
    for n = 2:N
        Y = ttm(Y, A[n], n)
    end
    return lamb, A, Y
end
