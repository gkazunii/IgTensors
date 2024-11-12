using Combinatorics
"""
    get_eta_from_tensor(T)
Get eta parameters of given normalized tensor T

# Examples
```julia-repl
julia> T = normalize(ones(3,3),1);
julia> get_eta_from_tensor(T)
[-2.197, 0.0, 0.0 ;
    0.0, 0.0, 0.0 ;
    0.0, 0.0, 0.0 ]
```

See also
http://nbviewer.org/gist/antimon2/8b156012fad69fcbc004ca0673a227d5/get_%CE%B7.jl.ipynb
"""
function get_eta_from_tensor(T)
    D = ndims(T)
    eta = copy(T)
    etap = reshape(@view(eta[end:-1:1]), size(eta))
    for d = 1:D
        cumsum!(etap,etap, dims=d)
    end
    eta
end

"""
    get_tensor_from_theta(theta)
Get tensor from given theta parameter theta
"""
function get_tensor_from_theta(theta)
    D = ndims(theta)
    tensor = similar(theta)
    tensor = cumsum(theta, dims=1)
    for d = 2:D
        cumsum!(tensor,tensor,dims=d)
    end
    exp.(tensor)
end

"""
    get_theta_from_tensor(T)
Get theta from given tensor T.
The computation include Logrithm.
If input tensor T contains very small values, result
include NaN. To avoid this situtaion, once you set
`avoid_vanish==true`, all element are added `eps_val`
get_theta_from_tensor( get_tensor_from_theta( tehta ) ) == theta
"""
function get_theta_from_tensor(T; avoid_vanish = false, add_val = 1.0e-3)
    #provided by
    #https://stackoverflow.com/questions/74813065/inverse-of-cumsum-in-julia
    decumsum(Y; dims) = [Y[I] - (
        I[dims]==1 ? 0 : Y[(ifelse(k == dims,I[k]-1,I[k])
          for k in 1:ndims(Y))...]
        ) for I in CartesianIndices(Y)]

    D = ndims(T)

    if avoid_vanish
        T .+= add_val
    end

    logT = log.(T)
    theta = decumsum(logT, dims=1)
    for d = 2 : D
        theta = decumsum(theta, dims=d)
    end
    return theta
end

"""
    get_tensor_from_eta(eta)
Get theta from given tensor T.
get_eta_from_tensor( get_tensor_from_eta( eta ) ) == eta
"""
function get_tensor_from_eta(eta)
    #provided by
    #https://stackoverflow.com/questions/74813065/inverse-of-cumsum-in-julia
    decumsum(Y; dims) = [Y[I] - (
        I[dims]==1 ? 0 : Y[(ifelse(k == dims,I[k]-1,I[k])
          for k in 1:ndims(Y))...]
        ) for I in CartesianIndices(Y)]

    etap = reverse(eta)
    D = ndims(eta)
    tensor = decumsum(etap, dims=1)
    for d = 2 : D
        tensor = decumsum(tensor, dims=d)
    end
    return reverse(tensor)
end

function get_eta_B(eta,M,dt)
    n_params = length(M)
    eta_vec = Vector{dt}(undef,n_params)
    get_eta_B!(eta_vec,eta,M)
end


function get_eta_B!(eta_vec,eta,M)
    n_params = length(M)
    for u = 1:n_params
        idx = M[u]
        eta_vec[u] = eta[idx]
    end
    eta_vec
end

function get_theta_B!(theta_vec,theta,M)
    n_params = length(M)
    for u = 1:n_params
        idx = M[u]
        theta_vec[u] = theta[idx]
    end
    theta_vec
end

function get_theta!(theta, theta_B, M)
    #theta .= 0.0
    n_params = length(M)
    for u = 1:n_params
        idx = M[u]
        theta[idx] = theta_B[u]
    end
    theta
end


"""
function get_theta_from_tensor2(T)
    D = ndims(T)
    mbss = zeros( [2 for d = 1:D]... )
    mbs = +1
    for d = 0 : D
        for col in collect(combinations(1:D,d))
            idx = [j in col ? 1 : 2 for j = 1:D]
            mbss[idx...] = mbs
        end
        mbs *= -1
    end

    theta = similar( T )
    idx111 = CartesianIndices(T)[1]
    idx222 = CartesianIndices(mbss)[end]
    #@show idx222
    logT = log.(T)
    for idx in CartesianIndices(T)
        idx_begin = max( idx - idx111, idx111)
        #@show idx, idx - idx_begin, idx222-(idx-idx_begin):idx222
        ibsr = idx222-(idx-idx_begin):idx222
        theta[idx] = sum(mbss[ibsr] .* logT[ idx_begin : idx ])
    end
    return theta
end
"""

function get_Tt_from_theta!(Tt, theta; avoid_vanish = false, upper_limit = 80)
    D = ndims(theta)
    cumsum!(Tt,theta,dims=1)
    for d = 2:D
        cumsum!(Tt,Tt,dims=d)
    end
	if avoid_vanish
		println("cut off large theta")
		Tt[ Tt.>upper_limit ] .= upper_limit #.* rand(0.9:1.1)
	end
    Tt .= exp.(Tt)
end
