"""
TMacTT
See Algorithm 2 in the
[original paper]()

# Aruguments
- 'X' : input tensor
- 'W' : binary tensor if X_ijk is missing then W_ijk = 0, otherwise 1
- 'alpha' : hyper parameter
- 'r' : hyper parameter, rank, vector
- 'Xgt' : ground truth tensor for printing verbose
"""
function TMacTT(X, W, r;alpha=NaN, iter_max=1000, verbose=true, Xgt=NaN, verbose_inv=5, tol=1.0e-4, avoid_singular=true, small_number=eps())
    idxs_missing = findall( W .== 0 )
    D = ndims(X)
    sizeX = size(X)
    normT = norm(X)

    if !(alpha isa Array)
        delta = zeros(D-1)
        alpha = zeros(D-1)
        for d=1:D-1
            delta[d] = min( prod(sizeX[1:d]), prod(sizeX[d+1:D]))
        end
        for d=1:D-1
            alpha[d] = delta[d]/sum(delta)
        end
    end

    Xs = Vector{Matrix{Float64}}(undef,D-1)
    Us = Vector{Matrix{Float64}}(undef,D-1)
    Vs = Vector{Matrix{Float64}}(undef,D-1)

    Xs_new = Vector{Matrix{Float64}}(undef,D-1)
    Us_new = Vector{Matrix{Float64}}(undef,D-1)
    Vs_new = Vector{Matrix{Float64}}(undef,D-1)
    for d = 1:D-1
        m = prod( sizeX[1:d] )
        n = prod( sizeX[d+1:D] )
        Us[d] = rand(m,r[d])
        #Vs[d] = rand(n,r[d])
        Vs[d] = rand(r[d],n)
    end

    X_pre = zeros( sizeX )
    for iter = 1:iter_max
        for d = 1:D-1
            Xs[d] = reshape(X, (prod(sizeX[1:d]),:)) #unfolding X on mode-d, matricization

            Us_new[d] = Xs[d] * (Vs[d])'
            #Vs_new[d] = pinv(Us_new[d]' * Us_new[d]) * Us_new[d]' * Xs[d]
            if avoid_singular
                Vs_new[d] = (Us_new[d]' * Us_new[d] + small_number * I ) \ (Us_new[d]') * Xs[d]
            else
                Vs_new[d] = (Us_new[d]' * Us_new[d] ) \ (Us_new[d]') * Xs[d]
            end

            Xs_new[d] = Us_new[d] * Vs_new[d]
        end

        foldX = zeros(size(X))
        for d = 1:D-1
            foldX .+= alpha[d] .* reshape(Xs_new[d], sizeX)
        end
        X[idxs_missing] .= foldX[idxs_missing]

        Vs = Vs_new
        Us = Us_new

        if iter > 1
            #diff_X = norm( X[idxs_missing] .- X_pre[idxs_missing] ) / length(idxs_missing)
            diff_X = norm( X .- X_pre ) / norm(X_pre)

            if verbose && iter > 1
                if mod(iter, verbose_inv) == 0
                    rms = norm(X .- Xgt)/norm(Xgt)
                    @show (iter, diff_X, rms)
                end
            end

            if diff_X < tol
                return X
            end
        end
        #X_pre[idxs_missing] .= X[idxs_missing]
        X_pre .= X

        """
        if verbose
            if iter % 100 == 0
                rms = norm(X .- Xgt)/norm(Xgt)
                @show (iter, rms)
            end
        end
        """
        if iter == iter_max
            println("TMacTT was not converged")
        end
    end
    return X
end
