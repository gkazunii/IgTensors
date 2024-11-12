using Combinatorics
using IterTools

function get_n_params_from_pos_theta(pos_theta)
    return sum(pos_theta) - 1
end

function get_n_params_from_m(m, J)
    D = length(J)
    if m == "cyc"
        intracts = get_intracts_for_cyc_2_body_approximation(D)
    else
        intracts = get_intracts_for_m_body_approximation(m, D)
    end
    return get_n_params_from_intracts(intracts, J)
end

function get_n_params_from_intracts(intracts, J)
    D = length(intracts)
    n_params = 1 # for zero-body parameter
    for d = 1:D
        d_body_intracts = intracts[d]
        #possible_d_body_pairs = collect(combinations([1:D;],d))
        possible_d_body_pairs = collect(combinations(1:D,d))
        D_C_d = binomial(D,d)
        for m = 1:D_C_d
            if d_body_intracts[m] == 0
                continue
            else
                n_params += prod( J[possible_d_body_pairs[m]] .-1 )
            end
        end
    end
    # We do not count zero-body theta as a parameter
    return n_params - 1
end

function get_M(intracts, J)
    D = length(intracts)
    n_params = get_n_params_from_intracts(intracts, J)
    M = Vector{CartesianIndex{D}}(undef,n_params)

    # We do not count zero-body theta as a parameter
    #p = 1
    #M[p] = CartesianIndex(ones(Int64,D)...)
    p = 1
    for d = 1 : D
        d_body_intracts = intracts[d]
        possible_d_body_pairs = collect(combinations(1:D,d))
        D_C_d = binomial(D,d)
        for m = 1:D_C_d
            if d_body_intracts[m] == 0
                continue
            else
                tp = (2:J[possible_d_body_pairs[m]][k] for k = 1:d)
                idx = ones(Int64,D)
                for over_write_idxs in product(tp...)
                    idx[possible_d_body_pairs[m]] .= over_write_idxs
                    M[p] = CartesianIndex(idx...)
                    p +=1
                end
            end
        end
    end

    @assert p - n_params == 1 "M has undef idx"
    return M
end

function get_M(pos_theta)
    n_params = get_n_params_from_pos_theta(pos_theta)
    D = ndims(pos_theta)
    M = Vector{CartesianIndex{D}}(undef,n_params)

    p = 1
    # We do not count zero-body theta as a parameter
    for idx in CartesianIndices(pos_theta)[2:end]
        if pos_theta[idx] == 1
            M[p] = idx
            p += 1
        end
    end
    return M
end
