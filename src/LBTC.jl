using StatsBase
using Plots
using Images
using FileIO
using Distributions
using Random
include("decomp.jl")

function LBTC!(T, W, m::Int;
        verbose=true, freq_verbose=1, tmax=100, tol=1.0e-3, newton_MBA=true,
        verbose_MBA=false, tmax_MBA=200, tol_MBA=1.0e-5)

    D = ndims(T)
    intracts = get_intracts_for_m_body_approximation(m,D)
    LBTC!(T, W, intracts;verbose=verbose,tmax=tmax,Tgt=Tgt,tol=tol, newton_MBA=newton_MBA,
            verbose_MBA=verbose_MBA, tmax_MBA=tmax_MBA, tol_MBA=tol_MBA)
end

function LBTC!(T, W, intracts;
        verbose=true, freq_verbose=1, tmax=80, tol=1.0e-3, newton_MBA=true,
        verbose_MBA=false, tmax_MBA=200, tol_MBA=1.0e-5)

    J = size(T)
    idx_missing = findall( W .== 0 )
    
    T[ idx_missing ] .= rand(Normal(50,10), length(idx_missing));
   
    normT = norm(T)
    res_pre = 0.0
    for t = 1 : tmax
        # #############
        # m-step
        # #############

        #normalize!(T, 1)
        Tm, _, _ = manybody_app(T, intracts,
                                verbose=verbose_MBA, tmax=tmax_MBA, newton=newton_MBA,
                                error_tol=tol_MBA)
        #normalize!(Tm, 1)
        res = norm(Tm .- T) / normT

        # #############
        # e-step
        # #############
        T[idx_missing] .= Tm[idx_missing] 

        if verbose && mod(t, freq_verbose) == 0
            @printf("Step:%2d error:%5f\n",t, res_pre - res)
        end

        if (res_pre - res) < tol && t > 1
            break
        end
        if t == tmax
            println()
            println("LBTC was not converged")
        end
        res_pre = res
        normT = norm(T)
    end
    return T
end
