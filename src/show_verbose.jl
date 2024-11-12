using Printf

function show_progress(t,res,newton,freq_verbose)
    if t == 1
        @printf("%10s %10s \n", "iter", "res")
    end
    if newton
        @printf("%10s %.8f \n", t,res)
    else
        if mod(t,freq_verbose) == 0
            @printf("%10s %.8f \n", t,res)
        end
    end
end

function show_conditions_decomp(J, intracts, n_params, newton, error_tol, inv_method, dt)
    println("--------------------------")
    if typeof(intracts) != Float64
        println("Many body Approximation is running...")
    else
        println("Legednre Decomposition is running...")
    end

    if newton == false
        opt_method = "grad"
    else
        opt_method = "newton"
    end
    size_of_input = prod(J)
    println("Optimization $opt_method")
    if newton == true
        println("How to get inv? $inv_method")
        if inv_method == "pinv"
            println("Pinv method takes time and memory.")
        end
    end

    println("error tol $error_tol")
    println("Shape of input tensor $J")
    println("Size of input tensor $size_of_input")
    println("Number of params $n_params")
    println("Size of FIM $n_params * $n_params")
    println("Data Type $dt")
    println("Activated intaraction:")
    println()
    if typeof(intracts) != Float64
        intracts_for_display = get_list_of_activated_intracts(intracts)
        display(intracts_for_display)
        println()
    end
end

function show_conditions_balancing(J, sf, n_params, newton, error_tol, inv_method, dt)
    println("--------------------------")
    println("$sf balancing is running...")

    if newton == false
        opt_method = "grad"
    else
        opt_method = "newton"
    end
    size_of_input = prod(J)
    println("Optimization $opt_method")
    if newton == true
        println("How to get inv? $inv_method")
        if inv_method == "pinv"
            println("Pinv method takes time and memory.")
        end
    end

    println("error tol $error_tol")
    println("Shape of input tensor $J")
    println("Size of input tensor $size_of_input")
    println("Number of params $n_params")
    println("Size of FIM $n_params * $n_params")
    println("Data Type $dt")
end
