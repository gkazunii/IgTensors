"""
SiLRTC: Simple Low Rank Tensor Completion
See Algorithm 1 in the
[original paper](https://ieeexplore.ieee.org/document/6138863)

# Aruguments
- 'X' : input tensor
- 'W' : binary tensor if X_ijk is missing then W_ijk = 0, otherwise 1
- 'tau' : hyper parameter
- 'Xgt' : ground truth tensor for printing verbose
"""
function SiLRTC(X, W; beta=NaN, tau=1.0e-2, 
        iter_max=100, verbose=false,
        Xgt=NaN, verbose_inv=5, tol=1.0e-3)
    
    idxs_missing = findall( W .== 0 )
    D = ndims(X)
    
    if !(beta isa Array)
        beta = ones(D)
    end
    
    sizeX = size(X)
    sum_beta = sum(beta)
    M = Vector{Matrix{Float64}}(undef,D)
    
    X_pre = zeros(sizeX)
    for iter = 1:iter_max
        diff_X = norm( X_pre .- X ) / norm(X_pre)
        # ##################
        # update matrices M
        # ##################

        for d = 1 : D
            Xd = tenmat(X,d) #unfolding X on mode-d, matricization
            M[d] = Dtau(Xd, tau)
        end

        foldM = zeros(size(X))
        for d = 1:D
            foldM .+= ( beta[d] .* matten( M[d], d, [size(X)...] ))
        end
        foldM = foldM ./ sum_beta

        X[idxs_missing] .= foldM[idxs_missing]
        
        if iter > 1
            diff_X = norm( X_pre .- X ) / norm(X_pre)
            if verbose
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
            if iter == iter_max
                println("SiLRTC was not converged")
            end
        end
        X_pre .= X
    end
    
    return X
end


"""
function main()
    r = 10
    Xgt = get_low_tucker_tensor([50,50,50],[r,r,r])
    W = generate_weight(Xgt, sr=30)
    Xin = init_missing_val!(deepcopy(Xgt), W)
    for tau in [10,100,1000]
        Xpre = SiLRTC(deepcopy(Xin), W, tau, iter_max=2000, verbose=true, Xgt=Xgt)
        @show (r, tau, RSE(Xgt,Xpre))
    end
end
"""