using Printf

"""
SiLRTCTT: Simple Low Rank Tensor Completion with Tucker
See Algorithm 1 in the
[original paper]()

# Aruguments
- 'X' : input tensor
- 'W' : binary tensor if X_ijk is missing then W_ijk = 0, otherwise 1
- 'alpha' : hyper parameter, D-1 dim vector
- 'beta' : hyper parameter, D-1 dim vector
- 'f' : hyper parameter, non-negative real value
- 'Xgt' : ground truth tensor for printing verbose
"""
function SiLRTCTT!(X, W;f=0.01, alpha=NaN, beta=NaN, iter_max=1000, verbose=false, verbose_inv=5, Xgt=NaN,tol=1.0e-4)
    idxs_missing = findall( W .== 0 )
    D = ndims(X)
    sizeX = size(X)
    normT = norm(X)

    if !( alpha isa Array && beta isa Array )
        beta = ones(D-1)
        delta = zeros(D-1)
        alpha = zeros(D-1)
        for d=1:D-1
            delta[d] = min( prod(sizeX[1:d]), prod(sizeX[d+1:D]))
        end
        for d=1:D-1
            alpha[d] = delta[d]/sum(delta)
            beta[d] = f*alpha[d]
        end
    end

    sum_beta = sum(beta)
    M = Vector{Matrix{Float64}}(undef,D)
    X_pre = zeros( sizeX )
    for iter = 1:iter_max

        # ##################
        # update matrices M
        # ##################

        for d = 1 : D-1
            Xd = reshape(X, (prod(sizeX[1:d]),:)) #unfolding X on mode-d, matricization
            tau = alpha[d] / beta[d]
            M[d] = Dtau(Xd, tau)
        end

        foldM = zeros(size(X))
        for d = 1:D-1
            #foldM .+= ( beta[d] .* matten( M[d], d, [size(X)...] ))
            foldM .+= ( beta[d] .* reshape( M[d], size(X) ))
        end
        foldM .= foldM ./ sum_beta

        X[idxs_missing] .= foldM[idxs_missing]
        """
        if verbose
            if iter % verbose_inval == 0
                rms = norm(X .- Xgt)/norm(Xgt)
                @show (iter, rms)
            end
        end
        """

        if iter > 1
            #diff_X = norm( X[idxs_missing] .- X_pre[idxs_missing] ) / length(idxs_missing)
            diff_X = norm( X .- X_pre ) / norm(X_pre)

            if verbose && iter > 1
                if mod(iter,verbose_inv) == 0
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
        end
        #X_pre[idxs_missing] .= X[idxs_missing]
        X_pre .= X
        
        if iter == iter_max
            println("SiLRTCTT was not converged")
        end
    end
    return X
end
