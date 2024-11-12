include("get_msk.jl")
include("manage_intracts.jl")
include("get_params.jl")
include("get_FIM.jl")
include("trivials.jl")
include("loader.jl")
include("show_verbose.jl")
include("project.jl")
include("trivials.jl")
using LinearAlgebra
using DoubleFloats
using Printf
using Quadmath
using Distributions

kl(P,Q) = sum( P .* log.( P ./ Q ) ) - sum(P) + sum(Q)

"""
    manybody_app(T, m)
Conduct m-body approximation for given tensor T.
If m==1, the output is rank-1 approximation.
If m==D, the output is the same as input where D == ndims(T).
If m==0, the output is the uniform tensor.
...
# Arguments
- `T` : input tensor
- `m::Int` : use up to m-th order interactions.
...

"""
function manybody_app(T, m::Int; tmax=150, error_tol=1.0e-5, lr=0.01, newton=true, verbose=false, freq_verbose=200, inv_method="normal", dt=Float64)
    D = ndims(T)
    if m == D
        return T, 0, 0
    end
    intracts = get_intracts_for_m_body_approximation(m,D)
    manybody_app(T, intracts; tmax=tmax, error_tol=error_tol, lr=lr, newton=newton, verbose=verbose, freq_verbose=freq_verbose, inv_method=inv_method, dt=dt)
end

function manybody_app(T, m::String; tmax=150, error_tol=1.0e-5, lr=0.01, newton=true, verbose=false, freq_verbose=200, inv_method="normal", dt=Float64)
    D = ndims(T)
    if m == "cyc"
        intracts = get_intracts_for_cyc_2_body_approximation(D)
    else
        error("$m is not defined")
    end
    manybody_app(T, intracts; tmax=tmax, error_tol=error_tol, lr=lr, newton=newton, verbose=verbose, freq_verbose=freq_verbose, inv_method=inv_method, dt=dt)
end

"""
    manybody_app(T, intracts)
Conduct many-body approximation with intracts for given tensor T.
...
# Arguments
- `T` : input tensor
- `intracts` : binary array indicating intractions.
...

# Examples
```julia-repl
julia> T = rand(3,3,3,3);
# define intraction with all one-body interactions and
# two-body interactions of (1,2) and (1,4) and
# three-body interactions of (1,2,3).
julia> intracts = [ [1,1,1,1],[1,0,1,0,0,0],[1,0,0,0],[0] ];
julia> approximated_T, theta = manybody_app(T, intracts)
```

"""
function manybody_app(T, intracts; tmax=150, error_tol=1.0e-5, lr=0.01, newton=true, verbose=false, freq_verbose=200, inv_method="normal", dt=Float64)
    D = ndims(T)
    J = size(T)
    
    
    check_intracts(intracts, D)

    sum_input = sum(T)
    # if you want to comput more faster, use normalize!(T,1) instead of T = normalize(T,1)
    T = normalize(T,1)

    n_params = get_n_params_from_intracts(intracts, J)
    M = get_M(intracts, J)
    if verbose
        show_conditions_decomp(J, intracts, n_params, newton, error_tol,inv_method,dt)
    end

    theta = zeros(dt, J...)
    eta_hat = get_eta_from_tensor(T) # tensor
    T, theta, eta = mproject(J,M,theta,eta_hat,tmax=tmax, error_tol=error_tol, lr=lr, newton=newton, verbose=verbose, freq_verbose=freq_verbose, inv_method=inv_method, dt=dt)
    return T .* sum_input, theta, eta
end

"""
    legendre_decomp(T, pos_theta)
Conduct Legendre decomposition for given tensor T.
pos_theta[1] need to be 1 since we need normalizer.
...
# Arguments
- `T` : input tensor
- `pos_theta` : binary tensor that indicates where \theta use.
...

# Examples
```julia-repl
julia> T = rand(3,3,3);
julia> pos_theta = rand(0:1,3,3,3); pos_theta[1] = 1;
julia> approximated_T, _ = legendre_decomp(T, pos_theta)
```

"""
function legendre_decomp(T, pos_theta; tmax=150, error_tol=1.0e-5, lr=0.01, newton=true, verbose=false, freq_verbose=200, inv_method="normal", dt=Float64)
    D = ndims(T)
    J = size(T)
    check_pos_theta(pos_theta, J)

    sum_input = sum(T)
    # if you want to comput more faster, use normalize!(T,1) instead of T = normalize(T,1)
    T = normalize(T,1)

    n_params = get_n_params_from_pos_theta(pos_theta)
    M = get_M(pos_theta)
    if verbose
        show_conditions_decomp(J, NaN, n_params, newton, error_tol,inv_method,dt)
    end

    eta_hat = get_eta_from_tensor(T) # tensor
    theta = zeros(dt, J...)
    T, theta = mproject(J,M,theta,eta_hat,tmax=tmax, error_tol=error_tol, lr=lr, newton=newton, verbose=verbose, freq_verbose=freq_verbose, inv_method=inv_method, dt=dt)
    return T .* sum_input, theta, eta
end
