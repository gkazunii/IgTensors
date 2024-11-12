include("get_FIM.jl")

function mproject(J, M, theta, eta_hat;T0=NaN, balancing=false, tmax=150, error_tol=1.0e-5, lr=0.01, newton=false, verbose=false, freq_verbose=200, inv_method="normal", dt=Float64)
    n_params = length(M)
    eta_hat_B = get_eta_B(eta_hat, M, dt) # vector

    # Initialize parameters
    eta_B = zeros(n_params)
    theta_B = zeros(n_params)

    # Initialize FIM
    G = Symmetric(Matrix{dt}(undef, n_params, n_params))

    # Initialize Prob (unifrom dist)
    Tt = ones(dt, J...) ./ prod(J)

    if balancing
        Tt .= T0
        get_theta_B!(theta_B,theta,M)
    end

    res_old = 0.0
    eta_t = get_eta_from_tensor(Tt)
    eta_B = get_eta_B(eta_t,M,dt)
    for t = 1:tmax
        """
        if t == 1
        else
            eta_t = get_eta_from_tensor(Tt)
            get_eta_B!(eta_B, eta_t, M)
        end
        """

        if newton
            update_G!(G,eta_t,M)
			 #if t < 5
			 #	G[diagind(G)] .+= 1.0e-12
            #end
			 if inv_method == "normal" || inv_method == "lu"
                theta_B .-= G \ (eta_B .- eta_hat_B)
            elseif inv_method == "qr"
                theta_B .-= qr(G) \ (eta_B .- eta_hat_B)
            elseif inv_method == "pinv"
                theta_B .-= pinv(G) * (eta_B .- eta_hat_B)
            elseif inv_method == "svd"
                theta_B .-= svd(G) \ (eta_B .- eta_hat_B)
            else
                error("inv method is undefined")
            end
        else
            theta_B .-= lr.*(eta_B .- eta_hat_B)
        end
        get_theta!(theta, theta_B, M)
        get_Tt_from_theta!(Tt, theta)

        normalize!(Tt,1)

        eta_t = get_eta_from_tensor(Tt)
        get_eta_B!(eta_B, eta_t, M)
        res = norm( eta_B .- eta_hat_B )

        """
        if res_old > eps() && res > res_old && t > 2
            break
        end
        """
        if isnan(res)
            println("Error. Cost becomes NaN")
            break
        end
        if res < error_tol && t > 2
            if verbose
                show_progress(t,res,newton,freq_verbose)
            end
            break
        else
            if verbose
                show_progress(t,res,newton,freq_verbose)
            end
            res_old = res
        end
        if t == tmax
            println("Many body approximtion was not converged")
        end
    end
    return Tt, theta, eta_t
end
