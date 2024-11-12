using LinearAlgebra
using TensorToolbox
using Tullio
using Distributions
using Printf

function get_low_tucker_tensor(J,R)
    G = rand(R...)
    D = length(J)
    U = Vector{Matrix{Float64}}(undef,D)
    for d = 1:D
        U[d] = ( -0.5 .+ rand(J[d],R[d]) )
    end
    T = ttm(G, [U...], [1:D;])
    T =  T .* ( prod(J) ./ sum(T))
    return T
end

function generate_weight(T;sr=30)
    prob = zeros(100)
    prob[1:sr] .= 1
    W = rand(prob, size(T))
    return W
end

function Dtau(X,tau)
    D = svd(X)
    return D.U * diagm( max.(D.S .- tau, 0) ) * D.Vt
end

function init_missing_val!(X,W)
    mu = sum( W .* X ) / sum(W)
    prob = Normal(mu, 1)
    idxs_missing = findall( W .== 0 )
    X[idxs_missing] = rand(prob,length(idxs_missing))
    return X
end

"""
TMacTT
See Algorithm 2 in the
[original paper]()

# Aruguments
- 'X' : input tensor
- 'W' : binary tensor if X_ijk is missing then W_ijk = 0, otherwise 1
- 'alpha' : hyper parameter
- 'beta' : hyper parameter
- 'Xgt' : ground truth tensor for printing verbose
"""
function TMacTT(X, W, alpha, r; iter_max=1000, verbose=true, Xgt=NaN, verbose_inval=5, tol=1.0e-4)
    idxs_missing = findall( W .== 0 )
    D = ndims(X)
    sizeX = size(X)
    normT = norm(X)

    """
    delta = zeros(D-1)
    alpha = zeros(D-1)
    for d=1:D-1
        delta[d] = min( prod(sizeX[1:d]), prod(sizeX[d+1:D]))
    end
    for d=1:D-1
        alpha[d] = delta[d]/sum(delta)
    end
    """

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
            Vs_new[d] = (Us_new[d]' * Us_new[d]) \ (Us_new[d]') * Xs[d]

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
            diff_X = norm( X[idxs_missing] .- X_pre[idxs_missing] ) / length(idxs_missing)
            #diff_X = norm( X .- X_pre ) / normT

            if verbose && iter > 1
                if mod(iter, verbose_inval) == 0
                    rms = norm(X .- Xgt)/norm(Xgt)
                    @show (iter, diff_X, rms)
                end
            end

            if diff_X < tol
                return X
            end
        end
        X_pre[idxs_missing] .= X[idxs_missing]

        """
        if verbose
            if iter % 100 == 0
                rms = norm(X .- Xgt)/norm(Xgt)
                @show (iter, rms)
            end
        end
        """
    end
    return X
end

"""
SiLRTCTT: Simple Low Rank Tensor Completion with Tucker
See Algorithm 1 in the
[original paper]()

# Aruguments
- 'X' : input tensor
- 'W' : binary tensor if X_ijk is missing then W_ijk = 0, otherwise 1
- 'alpha' : hyper parameter
- 'beta' : hyper parameter
- 'Xgt' : ground truth tensor for printing verbose
"""
function SiLRTCTT!(X, W, alpha, beta; iter_max=1000, verbose=false,verbose_inval=5, Xgt=NaN,tol=1.0e-4)
    idxs_missing = findall( W .== 0 )
    D = ndims(X)
    sizeX = size(X)
    normT = norm(X)

    """
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
    """

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
            diff_X = norm( X[idxs_missing] .- X_pre[idxs_missing] ) / length(idxs_missing)
            #diff_X = norm( X .- X_pre ) / normT

            if verbose && iter > 1
                if Xgt isa Array
                    rms = norm(X .- Xgt)/norm(Xgt)
                else
                    rms = NaN
                end
                @printf("%4d %5f %5f \n", iter, diff_X, rms)

                if mod(iter, verbose_inval) == 0
                    rms = norm(X .- Xgt)/norm(Xgt)
                    @show (iter, diff_X, rms)
                end
            end

            if diff_X < tol
                return X
            end
        end
        X_pre[idxs_missing] .= X[idxs_missing]
    end
    return X
end


"""
SiLRTC: Simple Low Rank Tensor Completion
See Algorithm 1 in the
[original paper]()

# Aruguments
- 'X' : input tensor
- 'W' : binary tensor if X_ijk is missing then W_ijk = 0, otherwise 1
- 'alpha' : hyper parameter, vector
- 'beta' : hyper parameter, vector
- 'Xgt' : ground truth tensor for printing verbose
"""
function SiLRTC!(X, W, alpha, beta; iter_max=1000, verbose=false, verbose_inval=5, Xgt=0, tol=1.0e-5)
    idxs_missing = findall( W .== 0 )
    D = ndims(X)

    sizeX = size(X)
    sum_beta = sum(beta)
    M = Vector{Matrix{Float64}}(undef,D)

    X_pre = zeros( sizeX )
    for iter = 1:iter_max

        # ##################
        # update matrices M
        # ##################

        for d = 1 : D
            Xd = tenmat(X,d) #unfolding X on mode-d, matricization
            tau = alpha[d] / beta[d]
            M[d] = Dtau(Xd, tau)
        end

        foldM = zeros(size(X))
        for d = 1:D
            foldM .+= ( beta[d] .* matten( M[d], d, [size(X)...] ))
        end
        foldM .= foldM ./ sum_beta
        X[idxs_missing] .= foldM[idxs_missing]

        if iter > 1
            diff_X = norm( X[idxs_missing] .- X_pre[idxs_missing] )

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
                @show iter
                return X
            end

            if iter == iter_max
                println("SiLRTC was not converged")
            end
        end
        X_pre[idxs_missing] .= X[idxs_missing]
    end
    """
        if verbose
            if iter % 200 == 0
                rms = norm(X .- Xgt)/norm(Xgt)
                @show (iter, rms)
            end
        end
    end
    """
    return X
end


"""
HaLRTC: High Accuracy Low Rank Tensor Completion
See Algorithm 4 in the
[original paper]()

# Aruguments
- 'X' : input tensor
- 'W' : binary tensor if X_ijk is missing then W_ijk = 0, otherwise 1
- 'alpha' : hyper parameter, vector. sum(alpha) should be 1.
- 'rho' : hyper parameter
- 'Xgt' : ground truth tensor for printing verbose
"""
function HaLRTC!(X, W, alpha;rho=1.0e-5, iter_max=1000, verbose=true, verbose_inval=5, Xgt=NaN, tol=1.0e-5)
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

    X_pre = zeros( J )
    for iter = 1:iter_max
        # #################
        # update M tensors
        # #################

        for d = 1:D
            Xd = tenmat(X,d)
            Ydd = tenmat(Y[d],d)
            M[d] = matten( Dtau( Xd .+ Ydd/rho, alpha[d]/rho ), d, [J...])
        end

        tmp = ( sum(M) .- sum(Y)./rho ) ./ D
        Xhat .= (1 .- W) .* tmp + W .* X

        # update Y tensors
        for d = 1:D
            Y[d] .-= rho.*( M[d] .- X )
        end

        if iter > 1
            diff_X = norm( X[idxs_missing] .- X_pre[idxs_missing] ) / length(idxs_missing)

            if verbose && iter > 1
                if mod(iter, verbose_inval) == 0
                    rms = norm(Xhat .- Xgt)/norm(Xgt)
                    @show (iter, diff_X, rms)
                end
            end

            if diff_X < tol
                return Xhat
            end
        end
        X_pre[idxs_missing] .= X[idxs_missing]
    end
    return Xhat
end

function main3()
    r = 5
    Xgt = get_low_tucker_tensor([15,15,15,15,15],[r,r,r,r,r])
    W = generate_weight(Xgt, sr=50)
    Xin = init_missing_val!(deepcopy(Xgt), W)
    @show norm(Xin-Xgt)
    for f in [0.01,0.05,0.1,0.5,1]
        Xpre = SiLRTCTT!(deepcopy(Xin), W, f, iter_max=2500, verbose=true, Xgt=Xgt)
        @show (r, f, RSE(Xgt,Xpre))
        R = [3,3,3,3,3]
        Xpre = TMacTT(deepcopy(Xin), R, iter_max=2500, verbose=true, Xgt=Xgt)
        @show (r, f, RSE(Xgt,Xpre))
    end
end

function run_TMacTT()
    r = 5
    Xgt = get_low_tucker_tensor([40,40,40,40],[r,r,r,r])

    sizeX = size(Xgt)
    D = ndims(Xgt)
    delta = zeros(D-1)
    alpha = zeros(D-1)
    for d=1:D-1
        delta[d] = min( prod(sizeX[1:d]), prod(sizeX[d+1:D]))
    end
    for d=1:D-1
        alpha[d] = delta[d]/sum(delta)
    end

    W = generate_weight(Xgt, sr=30)
    Xin = init_missing_val!(deepcopy(Xgt), W)
    R = [4,4,4,4,4]
    Xpre = TMacTT(deepcopy(Xin), W, alpha, R, iter_max=2500, verbose=true, verbose_inval=5, Xgt=Xgt)
    @show (r, R, RSE(Xgt,Xpre))
end

function run_SiLRTCTT()
    r = 3
    Xgt = get_low_tucker_tensor([30,30,30,30],[r,r,r,r])
    W = generate_weight(Xgt, sr=50)
    Xin = init_missing_val!(deepcopy(Xgt), W)
    D = ndims(Xgt)
    sizeX = size(Xgt)

    for f in [0.01,0.05,0.1,0.5, 1]
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

        Xpre = SiLRTCTT!(deepcopy(Xin), W, alpha,beta, iter_max=500, verbose_inval=25, verbose=true, Xgt=Xgt)
        @show (r, f, RSE(Xgt,Xpre))
    end
end

function run_SiLRTC()
    r = 8
    Xgt = get_low_tucker_tensor([15,15,15,15,15],[r,r,r,r,r])
    W = generate_weight(Xgt, sr=50)
    Xin = init_missing_val!(deepcopy(Xgt), W)
    D = ndims(Xgt)
    @show norm(Xin-Xgt)
    for gamma in [10,100,100,1000]
        alpha = ones(D)/D
        beta = ones(D)/gamma
        Xpre = SiLRTC!(deepcopy(Xin), W, alpha, beta, iter_max=2000, verbose=true, Xgt=Xgt)
    end
end


function run_HaLRTC()
    r = 2
    Xgt = get_low_tucker_tensor([50,50,50,50],[r,r,r,r])
    W = generate_weight(Xgt, sr=20)
    Xin = init_missing_val!(deepcopy(Xgt), W)
    @show norm(Xin-Xgt)
    #for rho in [1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9]
    for rho in [1.0e-8, 1.0e-9]
        D = ndims(Xin)
        alpha = ones(D)/D
        Xpre = HaLRTC!(Xin, W, alpha, rho=rho, iter_max=1000, verbose=true, verbose_inval=1, Xgt=Xgt)
        @show (r, rho, RSE(Xgt,Xpre))
    end

end
RSE(a,b) = norm(a - b) / norm(a)
#run_SiLRTC()

