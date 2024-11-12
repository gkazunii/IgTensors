using LinearAlgebra
using TensorToolbox
using Tullio
using Distributions
using Printf
using TransmuteDims
using Strided

function TCU(X,d,k)
    N = ndims(X)
    @assert d < N
    if d <= k
        a = k-d+1
    else
        a = k-d+1+N
    end
    @strided tmp = permutedims(X,circshift(1:N,-a+1))
    tenmat(tmp,row=1:d)
end

function invTCU(M,d,k, presize)
    N = length(presize)
    a = d<=k ? k-d+1 : k-d+1+N
    X = reshape(M,Tuple(circshift(collect(presize),1-a)))
    @strided permutedims(X,circshift(1:N,a-1))
end

function get_a(d,k,N)
    if d <= k
        return k - d + 1
    else
        return k - d + 1 + N
    end
end

function PTRCRW!(X, M, R, alpha, d;RW=true, iter_max=1000, verbose=true, verbose_inval=5, Xgt=NaN, tol=1.0e-3)
    idxs_missing = findall( M .== 0 )
    N = ndims(X)
    J = size(X)

    Xd = Vector{Matrix{Float64}}(undef,N)
    W = Vector{Matrix{Float64}}(undef,N)
    H = Vector{Matrix{Float64}}(undef,N)
    if RW
        B = Vector{Matrix{Float64}}(undef,N)
    end

    for k = 1:N
        a = get_a(d,k,N)
        if a == 1
            s1 = R[k]*R[N]
        else
            s1 = R[k]*R[a-1]
        end

        if a == 1
            s2 = prod(J[k+1:N])
        elseif k+1 <= a-1
            s2 = prod(J[k+1:a-1])
        else
            s2 = prod(vcat(J[k+1:N]...,J[1:a-1]...))
        end
        H[k] = rand( s1, s2 )

        if RW
            Wdk = TCU(M,d,k)
            omega_kd = sum( Wdk )
            beta = zeros(size(Wdk)[1])
            for i = 1:size(Wdk)[1]
                omega_kdi = sum( Wdk[i,:] )
                beta[i] = max( omega_kdi/omega_kd ,1.0e-16)
            end
            B[k] = diagm(beta)
        end
    end

    X_pre = zeros( J )
    for iter = 1:iter_max
        for k = 1:N
            if iter == 1
                Xd[k] = TCU(X,d,k)
                W[k] = Xd[k] * H[k]'
            else
                Xd[k] .= TCU(X,d,k)
                W[k] .= Xd[k] * H[k]'
            end

            if RW
                H[k] .= ( W[k]'*B[k]*W[k] ) \ W[k]' * B[k] * Xd[k]
            else
                H[k] .= ( W[k]'*W[k] ) \ W[k]' * Xd[k]
            end

            Xd[k] .= W[k]*H[k]
        end

        foldM = zeros(J)
        for k = 1:N
            foldM .+= ( alpha[k] .* invTCU(Xd[k],d,k,J) )
        end
        X[idxs_missing] .= foldM[idxs_missing]

        if iter > 1
            diff_X = norm( X_pre .- X ) / norm(X_pre)

            if verbose && iter > 1
                if mod(iter, verbose_inval) == 0
                    if Xgt isa Array
                        rms = norm(X .- Xgt)/norm(Xgt)
                    else
                        rms = NaN
                    end
                    @printf("%4d %5f %5f \n", iter, diff_X, rms)
                end
            end

            if diff_X < tol
                return X
            end

            if iter == iter_max
                println("PTRCRW was not converged")
            end
        end
        X_pre .= X
    end

    return X
end