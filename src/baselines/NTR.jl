using LinearAlgebra
using TensorToolbox
using StaticArrays

shiftdim(T) = permutedims(T,vcat([2:ndims(T);],[1]))
function Z_neq(Z,n)
    Z = circshift(Z,-n)
    N = length(Z)
    P = Z[1]
    for i=1:N-2
        zl = reshape(P,(:,size(Z[i])[3]))
        zr = reshape(Z[i+1],(size(Z[i+1])[1],:))
        P = zl*zr
    end
    Z_neq_out = reshape(P,(size(Z[1])[1],:,size(Z[N-1])[3]))
    return Z_neq_out
end

function NTR_MM(Y,r;lra=true,t_inner=100, verbose=false,ε=eps(),Tol=1e-4,MaxIter=400,ω=0.1,lra_parameter=Inf, LRA_R=20)
    n = size(Y)
    d = length(n)

    node = Vector{Array{Float64,3}}(undef,d)
    for i=1:d
        if i != d
            nodei = rand(r[i],n[i],r[i+1])
        else
            nodei = rand(r[d],n[d],r[1])
        end
        node[i] = nodei
    end

    od = [1:d;]
    err = 1.0
    od1 = od
    Y1 = Y
    s = undef

    Q = Array{Union{Float64,Matrix}}(undef,d)
    P = Array{Union{Float64,Matrix}}(undef,d)
    YY = Vector{Array{Float64,2}}(undef,d)
    for t=1:MaxIter
        Y = Y1
        err0 = err
        od = od1

        for i=1:d
            err0=err
            if i>1
                Y = shiftdim(Y)
                od = circshift(od,-1)
            end
            if t==1
                Y = reshape(Y,(n[od[1]],:))
                YY[i] = Y
                if lra
                    if n[i] > lra_parameter# || LRA_R >= size(YY[i])[1]
                        QQ = 1.0
                        PP = 1.0
                    else
                        QQ,_,_ = svd(Y)
                        if size(QQ)[2] >= LRA_R
                            QQ = QQ[:,1:LRA_R]
                        end
                        PP = QQ' * Y
                    end
                    Q[i] = QQ
                    P[i] = PP
                end
            end
            A = node[od[1]]
            A = permutedims(A,(2,3,1))
            A = reshape(A,(n[od[1]],:))
            B = Z_neq(node,od[1])
            B = permutedims(B,(1,3,2))
            B = reshape(B,(r[od[2]]*r[od[1]],:))

            Ω= eye(size(B)[1])*ω
            if !(lra)
                A = loop_MM(YY[i],A,B,Ω,t_inner)
            else
                if n[i] > lra_parameter || LRA_R >= size(YY[i])[1]
                    A = loop_MM(YY[i],A,B,Ω,t_inner)
                else
                    A = loop_LraMM(Q[i],P[i],A,B,Ω,t_inner)
                end
            end

            if i == d
                err1 = norm(YY[i]-A*B)
                err = err1 / norm(Y)
                if verbose
                    println("iter:$t \t err=$err")
                end
            end

            A = reshape(A,(n[od[1]],r[od[2]],r[od[1]]))
            A = permutedims(A, (3,1,2))
            s = norm(A[:],2)
            node[ od[1] ] = A ./ s

        end
        if abs(err0-err) <= Tol || err <= Tol
            break
        end
    end
    node[ od[1] ] .*= s
    return node
end

function NTR(Y,r;method="MU",t_inner=100, verbose=false,ε=eps(),Tol=1e-4,MaxIter=400,ω=0.1,lra_parameter=Inf,LRA_R=20)
    if method == "MM"
        return NTR_MM(Y,r;lra=false,t_inner=t_inner, verbose=verbose, ε=ε,Tol=Tol,MaxIter=MaxIter,ω=ω,lra_parameter
                     =lra_parameter, LRA_R=LRA_R)
    elseif method == "lraMM"
        return NTR_MM(Y,r;lra=true,t_inner=t_inner, verbose=verbose, ε=ε,Tol=Tol,MaxIter=MaxIter,ω=ω,lra_parameter=lra_parameter,LRA_R=LRA_R)
    end

    n = size(Y)
    d = length(n)

    node = Vector{Array{Float64,3}}(undef,d)
    for i=1:d
        if i != d
            nodei = rand(r[i],n[i],r[i+1])
        else
            nodei = rand(r[d],n[d],r[1])
        end
        node[i] = nodei
    end

    od = [1:d;]
    err = 1.0
    for i=1:(MaxIter*d)
        err0=err
        if i>1
            Y = shiftdim(Y)
            od = circshift(od,-1)
        end
        Y = reshape(Y,(n[od[1]],:))
        A = node[od[1]]
        A = permutedims(A,(2,3,1))
        A = reshape(A,(n[od[1]],:))
        B = Z_neq(node,od[1])
        B = permutedims(B,(1,3,2))
        B = reshape(B,(r[od[2]]*r[od[1]],:))

        if method == "MU"
            A .= loop_MU(Y,A,B,t_inner,ε)
        elseif method == "APG"
            A = loop_APG(Y,A,B,t_inner)
        elseif method == "HALS"
            A = loop_HALS(Y,A,B',t_inner,ε)
        else
            error("method error")
        end

        if mod(i,d) == 0
            err1 = norm(Y-A*B)
            err = err1 / norm(Y)
            if verbose
                println("iter:$(i/d) \t err=$err")
            end
        end

        A = reshape(A,(n[od[1]],r[od[2]],r[od[1]]))
        A = permutedims(A, (3,1,2))
        node[ od[1] ] = A

        if mod(i,d) == 0 && i > 1
            if abs(err0-err) <= Tol || err <= Tol
                break
            end
        end

        Y = reshape(Y, n[od])
    end
    return node
end

function loop_LraMM(Q,P,A,B,Ω,t_inner)
    B1 = B*B'
    B11 = eye(size(B1)[1]) / (Ω+B1)
    B12 = Ω-B1
    B1211 = B12*B11
    B211 = Q*(P*B'*B11)
    Z = similar(A)
    for t = 1:t_inner
        Z .= A ./ 2.0
        Z .= abs.(Z) * B1211 + B211
        A .= Z + abs.(Z)
    end
    return A
end

function loop_MM(Y,A,B,Ω,t_inner)
    B1 = B*B'
    B2 = Y*B'
    B11 = eye(size(B1)[1]) / (Ω+B1)
    B12 = Ω-B1
    B1211 = B12*B11
    B211 = B2*B11
    Z = similar(A)
    for t = 1:t_inner
        Z .= A ./ 2.0
        Z .= (abs.(Z) * B1211 + B211)
        A .= Z + abs.(Z)
    end
    return A
end

function loop_HALS(Y,A,B,t_inner,ε)
    J2 = size(A)[2]
    P = Y*B
    Q = B'*B
    for t = 1:t_inner
        for j = 1:J2
            A[:,j] = max.(ε,A[:,j] + (P[:,j]-A*Q[:,j])/max.(ε,Q[j,j]))
        end
    end
    return A
end

function loop_MU(Y,A,B,t_inner,ε)
    YBT = Y*B'
    BBT = B*B'
    for jj = 1:t_inner
        A .= A.*(YBT ./ max.(ε,A*BBT))
    end
    return A
end

function loop_APG(Y,A,B,Iter)
    BB = B*B'
    L = opnorm(BB,2)
    α= 1.0
    X = A
    X1 = BB/L
    Y1 = Y*B'/L
    A_old = A
    for k = 1:Iter
        A_old .= A
        α_old = α
        A .= max.(0.0, X-X*X1+Y1)
        α = (1+sqrt(4*α^2+1))/2
        X .= A .+ ((α_old-1)/α) .* (A .- A_old)
    end
    return A
end

outer(v...) = reshape(kron(reverse(v)...),length.(v))
function reconst_train(G)
    D = length(G)
    sizesG = size.(G)
    R = zeros(Int64,D)
    J = zeros(Int64,D)
    for d = 1:D
        R[d] = sizesG[d][1]
        J[d] = sizesG[d][2]
    end
    rs = ( (1:R[d]) for d in 1:D)
    
    term = zeros(Float64, J...)
    for r in product(rs...)
        v = Vector{Array{Float64}}(undef, D)
        for d = 1:D
            if d == 1
                v[d] = G[1][ 1, :, r[2]]
            elseif d == D
                v[d] = G[D][ r[D], :, 1]
            else
                v[d] = G[d][ r[d], :, r[d+1]]
            end
        end
        term += outer(v...)
    end

    return term
end
