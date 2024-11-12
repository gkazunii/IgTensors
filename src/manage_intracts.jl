function get_intracts_for_cyc_2_body_approximation(D)
    intracts = get_intracts_for_m_body_approximation(1,D)
    possible_2_body_pairs = collect(combinations(1:D,2))
    k = 1
    for pair in possible_2_body_pairs
        if pair[2] - pair[1] == 1
            intracts[2][k] = 1
        elseif pair[1] == 1 && pair[2] == D
            intracts[2][k] = 1
        end
        k += 1
    end
    return intracts
end

function get_intracts_for_m_body_approximation(m,D)
    if m == "cyc"
        intracts = get_intracts_for_cyc_2_body_approximation(D)
    else
        intracts = [ d <= m ? ones(Int64, binomial(D,d)) : zeros(Int64, binomial(D,d)) for d = 1:D]
    end
    return intracts
end

function check_pos_theta(pos_theta, J)
    D = length(J)
    @assert size(pos_theta) == J "the size of pos_theta should be same as T"
    @assert pos_theta[1] == 1 "zero_body theta have to be activated"
end

function check_intracts(intracts, D)
    @assert length(intracts) == D "Order of interaction should be $D"

    for d = 1:D
        D_C_d = binomial(D,d)
        @assert length(intracts[d]) == D_C_d "$d body interaction needs $D_C_d"
    end

    # Check Trivial Case
    total_num_intracts = sum( sum.(intracts) )
    if total_num_intracts == 2^D-1
        println("All interactions are activated.")
    end
    if total_num_intracts == 0
        println("Any interaction is activated.")
    end
end

function get_list_of_activated_intracts(intracts)
    D = length(intracts)
    intracts_for_display = Vector{Vector{Vector}}(undef,D)
    for d = 1:D
        possible_d_body_pairs = collect(combinations(1:D,d))
        D_C_d = binomial(D,d)
        number_d_body_intract = sum(intracts[d])
        if number_d_body_intract == 0
            intracts_for_display[d] = []
            continue
        end
        tmp_vec = Vector{Vector}(undef,number_d_body_intract)
        l = 1
        for m = 1:D_C_d
            if intracts[d][m] == 1
                tmp_vec[l] = possible_d_body_pairs[m]
                l += 1
            end
        end
        intracts_for_display[d] = tmp_vec
    end
    return intracts_for_display
end
