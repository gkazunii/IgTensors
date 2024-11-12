using LinearAlgebra
using InvertedIndices
include("show_verbose.jl")
include("project.jl")
include("manage_intracts.jl")
include("get_msk.jl")
include("get_params.jl")

function check_fiber(f,J)
    allequal(itr) = length(itr)==0 || all( isapprox(itr[1]), itr)
    D = length(J)
    @assert length(f) == D "length of fiber vector s need to be same as of input tensor"
    @assert allequal( sum.(f) ) == true "each sum in s[k] needs to be same"
    for d = 1:D
        @assert [size(f[d])...] == [J...][Not(d)] "size(s) need to be size(T)"
    end
end

function check_slice(s,J)
    allequal(itr) = length(itr)==0 || all( isapprox(itr[1]), itr)
    D = length(J)
    @assert length(s) == D "length of slice vector s need to be same as of input tensor"
    @assert allequal( sum.(s) ) == true "each sum in s[k] needs to be same"
    for d = 1:D
        @assert size(s[d])[1] == J[d] "size(s) need to be size(T)"
    end
end

"""
    slice_balancing(T)
Conduct slice balancing for tensor T.
Each sum for sliced T becomes 1 for normalized T.

# Examples
```julia-repl
# We normalize T for simplify.
julia> T = normalize(rand(3,3,3),1);
julia> Ts, _ = slice_balancing(T);
julia> sum(Ts,dims=[2,3])[:]
# [0.33 0.33 0.33]
julia> sum(Ts,dims=[1,3])[:]
# [0.33 0.33 0.33]
julia> sum(Ts,dims=[1,2])[:]
# [0.33 0.33 0.33]
```

"""
function slice_balancing(T; tmax=150, error_tol=1.0e-5, lr=0.01, newton=true, verbose=false, freq_verbose=200, inv_method="normal", dt=Float64)
    J = size(T)
    D = ndims(T)
    s = [ ones(J[d])/J[d] for d = 1:D]
    slice_balancing(T, s; tmax=tmax, error_tol=error_tol, lr=lr, newton=newton, verbose=verbose, freq_verbose=freq_verbose, inv_method=inv_method, dt=dt)
end

"""
    slice_balancing(T,s)
Conduct slice balancing for tensor T.
sum(T,dims=[1,..,k-1,k+1,..,D] becomes s[k] for normalized T.
Note that sum(s[1]) == sum(s[k]) should be satisfied for any k.

# Examples
```julia-repl
# We normalize T for simplify.
julia> T = normalize(rand(3,3,3),1);
julia> Ts, _ = slice_balancing(T);
julia> sum(Ts,dims=[2,3])[:]
# [0.33 0.33 0.33]
julia> sum(Ts,dims=[1,3])[:]
# [0.33 0.33 0.33]
julia> sum(Ts,dims=[1,2])[:]
# [0.33 0.33 0.33]
```

"""

function slice_balancing(T, s; tmax=150, error_tol=1.0e-5, lr=0.01, newton=true, verbose=false, freq_verbose=200, inv_method="normal", dt=Float64)
    D = ndims(T)
    J = size(T)
    sum_input = sum(T)
    check_slice(s,J)

    # if you want to comput more faster, use normalize!(T,1) instead of T = normalize(T,1)
    T = normalize(T,1)
    s = normalize.(s,1)

    intracts = get_intracts_for_m_body_approximation(1,D)
    M = get_M(intracts, J)

    eta_hat = zeros(J...)
    """
    We define balanced one-body eta
    """
    for d = 1:D
        eta_hat[(j == d ? (:) : 1 for j in 1:D)...] = reverse(cumsum(reverse(s[d])))
    end

    if verbose
        n_params = length(M)
        show_conditions_balancing(J, "fiber", n_params, newton, error_tol, inv_method, dt)
    end

    theta = get_theta_from_tensor(T)
    T, theta, eta = mproject(J,M,theta,eta_hat,T0=T, balancing=true,tmax=tmax, error_tol=error_tol, lr=lr, newton=newton, verbose=verbose, freq_verbose=freq_verbose, inv_method=inv_method, dt=dt)
    return T .* sum_input, theta, eta
end

"""
    fiber_balancing(T)
Conduct fiber balancing for tensor T.
Each sum for fiber T becomes 1 for normalized T.

# Examples
```julia-repl
# T do not have to be normalized but
# we normalize T for simplify.
julia> T = normalize(rand(3,3,3),1);
julia> Ts, _ = fiber_balancing(T);
julia> sum(Ts,dims=1)[1,:,:]
[[0.11 0.11 0.11]
 [0.11 0.11 0.11]
 [0.11 0.11 0.11]]
julia> sum(Ts,dims=2)[:,1,:]
[[0.11 0.11 0.11]
 [0.11 0.11 0.11]
 [0.11 0.11 0.11]]
julia> sum(Ts,dims=3)[:,:,1]
[[0.11 0.11 0.11]
 [0.11 0.11 0.11]
 [0.11 0.11 0.11]]
```

"""
function fiber_balancing(T; tmax=150, error_tol=1.0e-5, lr=0.01, newton=true, verbose=false, freq_verbose=200, inv_method="normal", dt=Float64)
    J = size(T)
    D = ndims(T)
    f = [ ones( [J...][Not(d)]... )/prod([J...][Not(d)]) for d = 1 : D ]
    fiber_balancing(T, f; tmax=tmax, error_tol=error_tol, lr=lr, newton=newton, verbose=verbose, freq_verbose=freq_verbose, inv_method=inv_method, dt=dt)
end

"""
    fiber_balancing(T, f)
Conduct fiber balancing for tensor T.
Each sum for fiber T becomes f for normalized T.
Note that f have to
For d=3,
sum(f[1],dims=1) == sum(f[2],dims=1)
sum(f[2],dims=2) == sum(f[3],dims=2)
sum(f[1],dims=2) == sum(f[3],dims=1)

# Examples
```julia-repl
# T do not have to be normalized but
# we normalize T for simplify.
julia> T = normalize(rand(2,2,2),1);
julia> f = [[1/4 1/4; 1/4 1/4], [0.3 0.2;0.2 0.3], [0.1 0.4;0.4 0.1]]
julia> Tf, _ = fiber_balancing(T,f);
julia> sum(Tf,dims=1)[1,:,:]
[[0.25 0.25]
 [0.25 0.25]]
julia> sum(Tf,dims=2)[:,1,:]
[[0.3 0.2]
 [0.2 0.3]]
julia> sum(Tf,dims=3)[:,:,1]
[[0.1 0.4]
 [0.4 0.1]]
```

"""
function fiber_balancing(T, f; tmax=150, error_tol=1.0e-5, lr=0.01, newton=true, verbose=false, freq_verbose=200, inv_method="normal", dt=Float64)
    D = ndims(T)
    J = size(T)
    sum_input = sum(T)
    check_fiber(f,J)

    # if you want to comput more faster, use normalize!(T,1) instead of T = normalize(T,1)
    T = normalize(T,1)
    f = normalize.(f,1)

    intracts = get_intracts_for_m_body_approximation(D-1,D)
    M = get_M(intracts, J)

    eta_hat = zeros(J...)
    """
    We define balanced m(<D)-body eta
    """
    for d = 1:D
        eta_hat[(j == d ? 1 : (:) for j in 1:D)...] .= reverse(get_cumsums_all_direction(f[d]))
    end

    if verbose
        n_params = length(M)
        show_conditions_balancing(J, "fiber", n_params, newton, error_tol, inv_method, dt)
    end

    theta = get_theta_from_tensor(T)
    T, theta, eta = mproject(J,M,theta,eta_hat,T0=T, balancing=true,tmax=tmax, error_tol=error_tol, lr=lr, newton=newton, verbose=verbose, freq_verbose=freq_verbose, inv_method=inv_method, dt=dt)
    return T .* sum_input, theta, eta
end

function get_cumsums_all_direction(a)
    D = ndims(a)
    a_sum = cumsum(a,dims=1)
    for d = 2:D
        cumsum!(a_sum,a_sum,dims=d)
    end
    return a_sum
end
