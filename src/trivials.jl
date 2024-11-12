using TensorToolbox
using LinearAlgebra
using InvertedIndices

"""
This function gives the best rank-1 approximation of the non-negative tensor `T`
"""
function best_rank1(T)
    input_tensor_depth = ndims(T)
    input_tensor_shape = size(T)

    partial_sums = []
    one_to_N = [1:input_tensor_depth;]
    for k=1:input_tensor_depth
        partial_sum = vec(sum(T, dims=one_to_N[Not(k)]))
        push!(partial_sums, partial_sum)
    end

    P = ttt(partial_sums[1], partial_sums[2])
    for n=3:input_tensor_depth
        P = ttt(P, partial_sums[n])
    end

    dev = sum(T)^(input_tensor_depth-1)
    P .= P ./ dev
    return P
end

"""
When all interactions are activated,
input tensor will be exactly reconstructed.
"""
function full_body_app(T)
    return T
end

"""
One-body approximation is equivalent with
the best rank-1 approximation minimizing KL divergence.
"""
function one_body_app(T)
    return best_rank1(T)
end

"""
Zero-body approximation always provides
uniform tensor.
"""
function zero_body_app(T)
    sum_input = sum(T)
    J = size(T)
    return ones(J...) ./prod(J) .* sum_input
end
