using Plots
function min_max_normalize(arr)
    max_arr = maximum(arr)
    min_arr = minimum(arr)
    diff_max_min = max_arr - min_arr
    normalize_arr = similar(arr)
    @show max_arr
    @show min_arr
    for k in CartesianIndices(arr)
        normalize_arr[k] = (arr[k]-min_arr) / diff_max_min
    end
    return normalize_arr
end

function get_gif(T, m, objn)
    J = size(T)
    D = ndims(T)
    n_img = J[end]
    if D == 4
        gif = rand(RGB{N0f8},J[1],J[2],n_img)
        for t = 1:n_img
            tensor_for_img = permutedims( T[:,:,:,t], [3,1,2] )
            imgc = colorview(RGB, tensor_for_img)
            imgc = map(clamp01nan,imgc)
            save_path = "../results/$objn/c/$m/$t.png"
            save(save_path, imgc)
            #println("saved $save_path")
            gif[:,:,t] .= map(clamp01nan, imgc)
        end
        save_path = "../results/$objn/c/$m/movie.gif"
        save(save_path, gif)
        println("saved $save_path")
    elseif D == 3
        gif = rand(Gray{N0f8},J[1],J[2],n_img)
        for t = 1:n_img
            mtx_for_img = T[:,:,t]
            imgg = colorview(Gray, mtx_for_img)
            imgg = map(clamp01nan,imgg)
            save_path = "../results/$objn/g/$m/$t.png"
            save(save_path, imgg)
            println("saved $save_path")
            gif[:,:,t] .= map(clamp01nan, imgg)
        end
        save_path = "../results/$objn/g/$m/movie.gif"
        save(save_path, gif)
        println("saved $save_path")
    end
end

function get_gif_params(xi,m,objn,te,cg)
    J = size(xi)
    if te == "theta"
        xi = min_max_normalize(xi)
        println("normalized")
        @show any(isnan.(xi))
    end
    n_img = J[4]
    gif = rand(RGB{N0f8},J[1],J[2],n_img)
    for t = 1:n_img
        tensor_for_img = permutedims( xi[:,:,:,t], [3,1,2] )
        imgc = colorview(RGB, tensor_for_img)
        imgc = map(clamp01nan,imgc)
        save_path = "../results/$objn/$cg/$m/$te/$t.png"
        save(save_path, imgc)
        #println("saved $save_path")
        gif[:,:,t] .= map(clamp01nan, imgc)
    end
    save_path = "../results/$objn/$cg/$m/$te/movie.gif"
    save(save_path, gif)
    println("saved $save_path")
end

function get_img(mtx,save_path; xlabel=false, ylabel=false, gx=false,gy=false)
    if ndims(mtx) == 1
        img = reshape(mtx,(:,1))
        imgg = Gray.(img)
        imgg = map(clamp01nan,imgg)
        plot(Gray.(imgg), xlabel=false, xticks=false,ylabel=ylabel)
    elseif ndims(mtx) == 2
        plot(Gray.(mtx), xlabel=xlabel, ylabel=ylabel)

        lw = 3
        ls = :dot
        J = size(mtx)
        xlim = J[1]
        tlim = J[2]

        a = 5
        b = 2
        if gx
            if occursin("orb1",save_path)
                len_orb = 0
                gtx1(t) = xlim/2 * sin(2*3.14*(t-len_orb)/tlim) + xlim/2
                plot!(gtx1,1,J[2],legend=false, xlim=(0,J[2]), ylim=(0,J[1]), lw=lw, ls=ls)
            elseif occursin("orb2",save_path)
                len_orb = 0
                gtx2(t) = xlim/2 * sin(2*3.14*(t-len_orb)/tlim) + xlim/2
                plot!(gtx2,1,J[2],legend=false, xlim=(0,J[2]), ylim=(0,J[1]), lw=lw, ls=ls)
            elseif occursin("orb3",save_path)
                len_orb = -10
                gtx3(t) = -(90/2)*(1/18) * ( 13 * cos(2*3.14*(t-len_orb)/60) - 5*cos(2*2*3.14*(t-len_orb)/60) - 2*cos(3*2*3.14*(t-len_orb)/60) - cos(4*2*3.14*(t-len_orb))/60 )+ 90/2
                plot!(gtx3,1,J[2],legend=false, xlim=(0,J[2]), ylim=(0,J[1]), lw=lw, ls=ls)
            elseif occursin("orb4",save_path)
                len_orb = 15
                gtx4(t) = xlim/10 * ( (a-b) * sin(4*3.14*(-t+len_orb)/tlim) - b * sin(4 * 3.14 *(-t+len_orb)/tlim *(a-b)/b )) + xlim/2
                plot!(gtx4,1,J[2],legend=false, xlim=(0,J[2]), ylim=(0,J[1]), lw=lw, ls=ls)
            else
                error("gtx(t) no defined")
            end
        end
        if gy
            if occursin("orb1",save_path)
                len_orb = 0
                gty1(t) = xlim/2 * cos(2*3.14/tlim*(t-len_orb)) + xlim/2
                plot!(gty1,1,J[2],legend=false, xlim=(0,J[2]), ylim=(0,J[1]), lw=lw, ls=ls)
            elseif occursin("orb2",save_path)
                len_orb = 0
                gty2(t) = xlim/2 * cos(2*2*3.14*(t-len_orb)/tlim) + xlim/2
                plot!(gty2,1,J[2],legend=false, xlim=(0,J[2]), ylim=(0,J[1]), lw=lw, ls=ls)
            elseif occursin("orb3",save_path)
                len_orb = -15
                gty3(t) = (90/2)*(1/16^3) * (16*sin(2*3.14*(-t-len_orb)/60))^3 + 90/2
                plot!(gty3,1,J[2],legend=false, xlim=(0,J[2]), ylim=(0,J[1]), lw=lw, ls=ls)
            elseif occursin("orb4",save_path)
                len_orb = -10
                gty4(t) = xlim/10 * ( (a-b) * cos(4*3.14*(-t-len_orb)/tlim) + b * cos(4 * 3.14 *(-t-len_orb)/tlim *(a-b)/b )) + xlim/2
                plot!(gty4,1,J[2],legend=false, xlim=(0,J[2]), ylim=(0,J[1]), lw=lw, ls=ls)
            else
                error("gty(t) no defined")
            end
        end
    elseif ndims(mtx) == 3
        # size(mtx) need to be (3,hoge,foo)
        cmtx = colorview(RGB,mtx)
        cmtx = map(clamp01nan,cmtx)
        plot(cmtx, xlabel=xlabel, ylabel=ylabel)
    end
    savefig(save_path)
    println("saved $save_path")
end

