function update_G!(G,eta,M)
    n_params = length(M)
    @inbounds for v = 1:n_params
        idxv = M[v]
        for u = 1:v
            idxu = M[u]
            idx = max(idxu,idxv)
            G.data[u,v] = eta[idx] - eta[idxu] * eta[idxv]
        end
    end
    return G
end
