function Dtau(X,tau)
    D = svd(X)
    return D.U * diagm( max.(D.S .- tau, 0) ) * D.Vt
end

"""
HaLRTC: High Accuracy Low Rank Tensor Completion
See Algorithm 4 in the
[original paper](https://ieeexplore.ieee.org/document/6138863)

# Aruguments
- 'X' : input tensor
- 'W' : binary tensor if X_ijk is missing then W_ijk = 0, otherwise 1
- 'rho' : hyper parameter
- 'Xgt' : ground truth tensor for printing verbose
"""
function HaLRTC(X, W;rho=1.0e-5, iter_max=100, tol=1.0e-3, verbose=true,Xgt=NaN,verbose_inv=5)
    idxs_missing = findall( W .== 0 )
    Xhat = X
    D = ndims(X)
    J = size(X)
    Y = Vector{Array{Float64,D}}(undef,D)
    M = Vector{Array{Float64,D}}(undef,D)
    for d = 1:D
        Y[d] = zeros( J )
        M[d] = zeros( J )
    end
    alpha = 1.0/D

    X_pre = zeros(J)
    for iter = 1:iter_max
        # #################
        # update M tensors
        # #################

        for d = 1:D
            Xd = tenmat(X,d)
            Ydd = tenmat(Y[d],d)
            M[d] = matten( Dtau( Xd .+ Ydd/rho, alpha/rho ), d, [J...])
        end

        tmp = ( sum(M) .- sum(Y)./rho ) ./ D
        Xhat .= (1 .- W) .* tmp + W .* X

        # update Y tensors
        for d = 1:D
            Y[d] .-= rho.*( M[d] .- X )
        end
        
        if iter > 1
            diff_X = norm( X_pre .- Xhat ) / norm(X_pre)
            if verbose
                if mod(iter,verbose_inv) == 0
                    if Xgt isa Array
                        rms = norm(Xhat .- Xgt)/norm(Xgt)
                    else
                        rms = NaN
                    end
                    @printf("%4d %5f %5f \n", iter, diff_X, rms)
                end
            end
            
            if diff_X < tol
                return Xhat
            end
            
            if iter == iter_max
                println("HaLRTC was not converged")
            end
        end
        X_pre .= Xhat
    end
    
    return Xhat
end