__precompile__()

module SmoothingSplines

import StatsBase: fit!, fit, RegressionModel, rle, ordinalrank, mean
using Reexport

export SmoothingSpline

@reexport using StatsBase

LAPACKFloat = Union{Float32,Float64}

include("matrices.jl")

type SmoothingSpline{T<:LAPACKFloat} <: RegressionModel
    Xorig::Vector{T} # original grid points, but sorted
    Yorig::Vector{T} # original values, sorted according to x
    Xrank::Vector{Int}
    Xdesign::Vector{T}
    Xcount::Vector{Int} # to how many observations does X correspond?
    Ydesign::Vector{T}
    weights::Vector{T} # don't implement this yet.
    RpαQtQ::Matrix{T} # in symmetric banded matrix format
    g::Vector{T} # fitted values
    γ::Vector{T} # 2nd derivatives of fitted vals
    λ::T
end

function fit{T<:LAPACKFloat}(::Type{SmoothingSpline}, X::AbstractVector{T}, Y::AbstractVector{T}, λ::T, wts::AbstractVector{T}=ones(Y))
    Xrank = ordinalrank(X) # maybe speed this up when already sorted
    Xperm = sortperm(X)
    Xorig = X[Xperm]
    Yorig = Y[Xperm]

    Xdesign, Xcount = rle(Xorig)
    ws = zeros(Xdesign)
    Ydesign = zeros(Xdesign)
    running_rle_mean!(Ydesign, ws, Yorig, Xcount, wts[Xperm])

    RpαQtQ = QtQpR(diff(Xdesign), λ, ws)
    pbtrf!('U', 2, RpαQtQ)

    spl = SmoothingSpline{T}(Xorig, Yorig, Xrank, Xdesign, Xcount, Ydesign, ws, RpαQtQ,
                zeros(Xdesign), zeros(T,length(Xdesign)-2), λ)
    fit!(spl)
end

function fit!{T<:LAPACKFloat}(spl::SmoothingSpline{T})
    Y = spl.Ydesign
    ws = spl.weights
    g = spl.g
    n = length(spl.g)
    h = diff(spl.Xdesign)
    λ = spl.λ
    Q = ReinschQ(h)

    RpαQtQ = spl.RpαQtQ
    γ = At_mul_B!(spl.γ, Q, Y) #Q^T*Y
    pbtrs!('U', 2, RpαQtQ, γ)
    A_mul_B!(g, Q, γ)
    broadcast!(/, g, g, ws)
    broadcast!(*, g, g, λ)
    broadcast!(-,g, Y, g)
    spl
end

function fit!{T<:LAPACKFloat}(spl::SmoothingSpline{T}, Y::AbstractVector{T})
    spl.Yorig = Y[spl.Xrank]
    fit!(spl)
end


function predict{T<:LAPACKFloat}(spl::SmoothingSpline{T})
    # need to convert back from RLE encoding
    Xcount = spl.Xcount
    curridx = 1::Int
    g = zeros(length(spl.Yorig))
    @inbounds for i=1:length(Xcount)
        @inbounds   for j=1:Xcount[i]
            g[curridx] = spl.g[i]
            curridx += 1
        end
    end
    g[spl.Xrank]
end

function predict{T<:SmoothingSplines.LAPACKFloat}(spl::SmoothingSpline{T}, x::T)
    n = length(spl.Xdesign)
    idxl = searchsortedlast(spl.Xdesign, x)
    idxr = idxl + 1
    if idxl == 0 # linear extrapolation to the left
        gl = spl.g[1]
        gr = spl.g[2]
        γ  = spl.γ[1]
        xl = spl.Xdesign[1]
        xr = spl.Xdesign[2]
        gprime = (gr-gl)/(xr-xl) - 1/6*(xr-xl)*γ
        val = gl - (xl-x)*gprime
    elseif idxl == n # linear extrapolation to the right
        gl = spl.g[n-1]
        gr = spl.g[n]
        γ  = spl.γ[n-2]
        xl = spl.Xdesign[n-1]
        xr = spl.Xdesign[n]
        gprime = (gr-gl)/(xr-xl) +1/6*(xr-xl)*γ
        val = gr + (x - xr)*gprime
    else # cubic interpolation
        xl = spl.Xdesign[idxl]
        xr = spl.Xdesign[idxr]
        γl = idxl == 1 ? zero(T) : spl.γ[idxl-1]
        γr = idxl == n-1 ? zero(T) : spl.γ[idxr-1]
        gl = spl.g[idxl]
        gr = spl.g[idxr]
        h = xr-xl
        val = ((x-xl)*gr + (xr-x)*gl)/h
        val -=  1/6*(x-xl)*(xr-x)*((1 + (x-xl)/h)*γr + (1+ (xr-x)/h)*γl)
    end
    val
end

function predict{T<:SmoothingSplines.LAPACKFloat}(spl::SmoothingSpline{T}, xs::AbstractVector{T})
    g = zeros(xs)
    for (i,x) in enumerate(xs)
        # can be made more efficient as in StatsBase ECDF code
        g[i] = predict(spl, x)
    end
    g
end

# update g and w in place
function running_rle_mean!{T<:Real}(g::AbstractVector{T}, w::AbstractVector{T}, Y::AbstractVector{T}, rlecount::AbstractVector{Int}, ws::AbstractVector{T})
  length(g) == length(rlecount) ||  throw(DimensionMismatch())
  length(Y) == length(ws) || throw(DimensionMismatch())
  curridx = 1::Int
  for i=1:length(rlecount)
    idxrange = curridx:(curridx+rlecount[i]-1)
    g[i] = mean(Y[idxrange], Weights(ws[idxrange])) #todo: use weights by default
    w[i] = sum(ws[idxrange])
    curridx += rlecount[i]
  end
  g
end


end # module
