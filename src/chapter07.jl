# Relevance Vector Machine (ARD sparse prior) for classification

struct RVModel{T<:Real}
    w::Array{T}
    α::Array{T}
    index::Vector{T}
end

#interface
function RVM(X::Matrix{T}, t::Vector{Int64}, α::Float64=1.0;
             kw...) where T<:Real
    n = size(X, 1)
    d = size(X, 2)
    size(t, 1) == n || throw(DimensionMismatch("Sizes of X and t mismatch."))

    K = size(unique(t))
    if K > 2
        α = ones(Float64, d, K) .* α
    elseif K == 2
        α = ones(Float64, d) .* α
    else
        throw(TypeError("Number of classes less than 2."))
    end
    RVM!(X, t, α; kw...)
end

# core algorithm
function RVM!(X::Matrix{T}, t::Vector{Int64}, α::Vector{Float64};
              tol::Float64=1e-4, maxiter::Int64=10000) where T<:Real
    n = size(X, 1)
    d = size(X, 2)
    size(t, 1) == n || throw(DimensionMismatch("Sizes of X and t mismatch."))
    size(α, 1) == d || throw(DimensionMismatch("Sizes of X and initial α mismatch."))

    # preallocate type-II likelihood (evidence) vector
    llh2 = Vector{T}(undef, maxiter)
    fill!(llh2, -Inf)
    # Hessian
    # pre-allocate memories
    a = Vector{T}(undef, n)
    w = zeros(T, d)
    h = ones(T, n)
    # begin iteration
    for iter = 2:maxiter
        ind = findall(1 ./ α .> tol) # index of nonzeros
        αtmp = copy(α[ind])
        wtmp = copy(w[ind])
        # find posterior w - mode and hessian
        H, llh2[iter] = Logit!(wtmp, αtmp, X[:, ind], t, tol, maxiter, a, h)
        #llh2[iter] = llh - n * log(2π) - dot(αtmp, wtmp .^ 2)
        w[ind] .= wtmp
        for j = 1:size(ind)
            @inbounds llh2[iter] += 0.5log(αtmp[j])
        end
        llh2[iter] -= 0.5log(det(H))
        if abs(llh2[iter] - llh2[iter-1]) < tol * abs(llh2[iter-1])
            return RVModel(w, α)
        end
        Σ = Hermitian(H) \ I
        for j = 1:size(ind)
            @inbounds αtmp[j] = (1 - αtmp[j]Σ[j, j]) / wtmp[j]^2
        end
        α[ind] .= αtmp
    end
    # construct the model
    warn("Not converged after", iter, " steps. Results may be inaccurate.")
    return RVModel(w, α)
end

function RVM!(X::Matrix{T}, t::Vector{Int64}, α::Matrix{Float64};
              tol=1e-4, maxiter=1000, method=:NewtonRaphsonBlock) where T<:Real
    # Multinomial
    n = size(X, 1)
    d = size(X, 2)
    size(t, 1) == n || throw(DimensionMismatch("Sizes of X and t mismatch."))
    size(α, 1) == d || throw(DimensionMismatch("Sizes of X and initial α mismatch."))
    K = size(unique(t))  # total number of classes
    size(α, 2) != K || throw(TypeError("Number of classes and size of α mismatch."))

    # initialise
    # preallocate type-II likelihood (evidence) vector
    llh2 = Vector{T}(undef, maxiter)
    fill!(llh2, -Inf)
    w = Vector{T}(undef, d, K)
    if method == :NewtonRaphsonBlock  # alternate for each class
        iter = 1
        t2 = similar(t); h = similar(t); a = similar(t)
        while (iter < maxiter + 1)
            for k = 1:K
                iter += 1
                w2 = @view w[:, k]
                α2 = @view α[:, k]
                fill!(t2, 0)
                t2[findall(x -> x==k, t)] = 1
                ind = findall(1 ./ α2 .> tol) # index of nonzeros
                αtmp = copy(α2[ind])
                wtmp = copy(w2[ind])
                # find posterior w - mode and hessian
                H, llh2[iter] = Logit!(wtmp, αtmp, X[:, ind], t2, tol, maxiter, a, h)
                #llh2[iter] = llh - n * log(2π) - dot(αtmp, wtmp .^ 2)
                for j = 1:size(ind)
                    @inbounds llh2[iter] += 0.5log(αtmp[j])
                end
                llh2[iter] -= 0.5log(det(H))
                Σ = Hermitian(H) \ I
                for j = 1:size(ind)
                    @inbounds αtmp[j] = (1 - αtmp[j]Σ[j, j]) / wtmp[j]^2
                end
                α2[ind] .= αtmp
                w2[ind] .= wtmp
            end
            # check convergence
            if abs(llh2[iter] - llh2[iter-1]) < tol * abs(llh2[iter-1])
                return RVModel(w, α)
            end
        end
        warn("Not converged after", iter, " steps. Results may be inaccurate.")
        return RVModel(w, α)

    elseif method == :NewtonRaphson  # update weights of each class simutenously
        Y = Matrix{T}(undef, n, K)
        logY = Matrix{T}(undef, n, K)
        for iter = 2:maxiter
            ind = unique!([item[1] for item in findall(1 ./ α .> tol)])
            αtmp = copy(α[ind, :])
            wtmp = copy(w[ind, :])
            n_ind = size(ind, 1)
            # find posterior mean and variance via Laplace approx.
            H, llh = Logit!(wtmp, αtmp, X[:, ind], t, tol, maxiter, Y, logY)
            newdk = size(H, 1)
            llh2[iter] = llh - 0.5log(det(H))
            for j = 1:newdk
                @inbounds llh2[iter] += 0.5log(αtmp[:][j])
            end
            w[ind, :] .= wtmp
            if abs(llh2[iter] - llh2[iter-1]) < tol * abs(llh2[iter-1])
                return RVModel(w, α)
            end
            Σ = Hermitian(H) \ I
            for k = 1:K, i=1:n_ind
                l = (k-1) * n_ind + i
                @inbounds αtmp[j, k] = (1 - αtmp[j, k]Σ[l, l]) / wtmp[j, k]^2
            end
            α[ind, :] .= αtmp
            #rand() < 0.1 ? α .+=
        end
        warn("Not converged after", iter, " steps. Results may be inaccurate.")
        return RVModel(w, α)
    end
end

function Logit!(w::Vector{T}, α::Vector{T}, X::Matrix{T},
                t::Vector{Int64}, tol::Float64, maxiter::Int64,
                a::Vector{T}, h::Vector{Int64}) where T<:Real
    n = size(X, 1)
    d = size(X, 2)
    H = Matrix{T}(undef, d, d)
    g = Vector{T}(undef, d)
    fill!(h, 1)
    h[findall(iszero, t)] = -1
    Xt = transpose(X)
    mul!(a, X, w)
    llhp = -Inf
    wp = similar(w)
    for iter = 2:maxiter
        y .= 1 ./ (1 .+ exp(-1 .* a))
        # update Hessian
        fill!(H, 0)
        for nn = 1:n
            H .+= Xt[:, nn] .* transpose(Xt[:, nn]) .* y[nn] .* (1 .- y[nn])
        end
        add_diagonal!(H, α)
        # update gradient
        mul!(g, Xt, y .- t)
        g .+= α .* w
        Δw = -Hermitian(H) \ g
        # update w
        copyto!(wp, w)
        w .+= Δw
        mul!(a, X, w)
        llh = -sum(log1p.(exp.(-h .* a))) - 0.5sum(α .* w .^ 2)
        while llh - llhp < 0
            Δw ./= 2
            w .= wp .+ Δw
            mul!(a, X, w)
            llh = -sum(log1p.(exp.(-h .* a))) - 0.5sum(α .* w .^ 2)
        end
        llh - llhp < tol ? break : llhp = llh
    end
    H, llh
end

function Logit!(w::Matrix{T}, α::Matrix{T}, X::Matrix{T},
                t::Vector{Int64}, tol::Float64, maxiter::Int64) where T<:Real
    n = size(t, 1)
    d = size(X, 2)
    K = size(unique(t), 1) # number of classes
    dk = d * K
    H = zeros(T, dk, dk)
    g = zeros(T, dk)
    A = Matrix{T}(undef, n, K); Y = similar(A); logY = similar(A)
    wp = similar(w)
    llhp = -Inf
    mul!(A, X, w)
    logY .= A .- log.(sum(exp.(A), dims=2))
    Y .= exp.(Y)
    for iter = 2:maxiter
        # update Hessian
        fill!(H, 0)
        for k = 1:K, i = 1:d
            idx = d * (k-1) + i
            for l = 1:K, j = 1:d
                idy = d * (l-1) + j
                k == l ? δ = 1 : δ = 0
                for nn = 1:n
                    @inbounds H[idy, idx] .+= X[nn, i] .* X[nn, j] .*
                                          Y[nn, k] .* (δ - Y[nn, l])
                end
            end
        end
        add_diagonal!(H, α[:])
        # update gradient
        fill!(g, 0)
        for k = 1:K, i = 1:d, nn = 1:n
            t[nn] == k ? δ = 1 : δ = 0
            @inbounds g[d * (k-1) + i] .+= (δ - Y[nn, k]) .* X[nn, i]
        end
        g .-= w[:] .* α[:]
        copyto!(wp, w)
        Δw = reshape(Hermitian(H) \ g, d, K)
        w .+= Δw
        mul!(A, X, w)
        logY .= A .- log.(sum(exp.(A), dims=2))
        Y .= exp.(Y)
        # update likelihood
        llh = -sum(0.5 .* α[:] .* w[:] .* w[:])
        for i = 1:n
            @inbounds llh += logY[i, t[i]]
        end
        # not really necessary
        while (llh - llhp < 0)
            Δw ./= 2
            w .= wp .+ Δw
            mul!(A, X, w)
            logY .= A .- log.(sum(exp.(A), dims=2))
            Y .= exp.(Y)
            llh = -sum(0.5 .* α[:] .* w[:] .* w[:])
            for i = 1:n
                @inbounds llh += logY[i, t[i]]
            end
        end
        abs(llh - llhp) < tol ? break : llhp = llh
    end
    return H, llh
end












function Logit!(w::Matrix{T}, α::Matrix{T}, X::Matrix{T},
                t::Vector{Int64}, tol::Float64, maxiter::Int64,
                block::Bool) where T<:Real
    n = size(X, 1)
    d = size(X, 2)
    K = size(unique(t), 1) # number of classes
    H = Matrix{T}(undef, d, d)
    g = Vector{T}(undef, d, d)
    for iter = 1:maxiter
        for k = 1:K
            h[findall(x -> x!=k, t)] = -1
            wtemp = @view w[:, k]
            αtemp = @view α[:, k]
            mul!(a, X, wtemp)
            y .= 1 ./ (1 .+ exp(-1 .* a))
            #r .= sqrt.(y .* (1 .- y))
            # update Hessian
            fill!(H, 0)
            for nn = 1:n
                H .+= Xt[:, nn] .* transpose(Xt[:, nn]) .* y[nn] .* (1 .- y[nn])
            end
            add_diagonal!(H, αtemp)
            cholesky!(H)
            # update gradient
            fill!(g, 0)
            for nn = 1:n
                t[nn] == k ? δ = 1 : δ = 0
                g .+= Xt[:, nn] .* (δ .- y[nn])
            end
            g .-= αtemp .* wtemp
            Δw = H \ (transpose(H) \ g)
            # update weights
            copyto!(wp, wtemp)
            wtemp .+= Δw
            mul!(a, X, wtemp)
            llh[iter] = -sum(log1p.(exp.(-h .* a))) - 0.5sum(αtemp .* wtemp .^ 2)
            while llh[iter] - llh[iter-1] < 0  # line search
                wtemp .= wp .+ (Δw ./ 2)
                mul!(a, X, wtemp)
                llh[iter] = -sum(log1p.(exp.(-h .* a))) - 0.5sum(αtemp .* wtemp .^ 2)
            end
        end
        if llh[iter] - llh[iter - 1] < tol
            # recalculate the H
            break
        end
    end
end
