using LinearAlgebra
using Random

function KL(A, B)
    n, m = size(A)
    kl = 0.0
    for i = 1:n
        for j = 1:m
            kl += A[i,j] * log( A[i,j] / B[i,j] ) - A[i,j] + B[i,j]
        end
    end
    return kl
end

function nmf_euc(X, r ; max_iter=200, tol = 0.0001)
    m, n = size(X)
    W = rand(m, r)
    H = rand(r, n)

    error_at_init = norm(X - W*H)
    previous_error = error_at_init
    for iter in 1:max_iter
        H .=  H .* ( W' * X ) ./ ( W' * W * H  )
        W .=  W .* ( X  * H') ./ ( W  * H * H' )

        if tol > 0 && iter % 10 == 0
            error = norm(X - W*H)
            if (previous_error - error) / error_at_init < tol
                break
            end
            previous_error = error
        end
    end

    return W, H
end

function nmf_kl(X, r ; max_iter=200, tol = 0.0001)
    m, n = size(X)
    W = rand(m, r)
    H = rand(r, n)
    one_mn = ones(m, n)

    error_at_init = KL(X, W*H)
    previous_error = error_at_init
    for iter in 1:max_iter
        println(KL(X, W*H))
        H .=  H .* ( W' * ( X ./ (W*H))) ./ ( W' * one_mn )
        W .=  W .* ( (X ./ (W*H) ) * H') ./ ( one_mn * H' )

        if tol > 0 && iter % 10 == 0
            error = KL(X, W*H)
            if (previous_error - error) / error_at_init < tol
                break
            end
            previous_error = error
        end
    end

    return W, H
end

function NNTF(A, r; max_iter=200, tol = 0.0001)
    C = A
    r0 = 1
    d = ndims(A)
    n = size(A)
    G = Vector{Array{Float64}}(undef,d)
    for i = 1:(d-1)
        if i == 1
            C = reshape(C, (r0*n[i],:))
        else
            C = reshape(C, (r[i-1]*n[i],:))
        end
        W, H = nmf_euc(C, r[i], max_iter=max_iter, tol=tol)
        H .= norm(W).* H
        W .= W ./ norm(W)
        if i == 1
            G[i] = reshape(W, (1, n[i], r[i]))
        else
            G[i] = reshape(W, (r[i-1], n[i], r[i]))
        end
        C = H
    end
    G[d] = reshape(C, (r[d-1],n[d],1))
   
    return G
end

outer(v...) = reshape(kron(reverse(v)...),length.(v))
function reconst_train(G)
    D = length(G)
    sizesG = size.(G)
    R = zeros(Int64,D)
    J = zeros(Int64,D)
    for d = 1:D
        R[d] = sizesG[d][1]
        J[d] = sizesG[d][2]
    end
    rs = ( (1:R[d]) for d in 1:D)
    
    term = zeros(Float64, J...)
    for r in product(rs...)
        v = Vector{Array{Float64}}(undef, D)
        for d = 1:D
            if d == 1
                v[d] = G[1][ 1, :, r[2]]
            elseif d == D
                v[d] = G[D][ r[D], :, 1]
            else
                v[d] = G[d][ r[d], :, r[d+1]]
            end
        end
        term += outer(v...)
    end

    return term
end

function get_n_params_train(R,J)
    n_params = 0
    D = length(J)
    for i = 1:D
        if i == 1
            n_params += 1*J[1]*R[1]
        elseif i == D
            n_params += R[D-1]*J[D]*1
        else
            n_params += R[i-1]*J[i]*R[i]
        end
    end
    return n_params
end