module SmoothingSplines

import StatsBase: fit!, fit, RegressionModel, rle, ordinalrank

export SmoothingSpline, fit!, fit, predict


LAPACKFloat = Union{Float32,Float64}

include("matrices.jl")

type SmoothingSpline{T<:LAPACKFloat} <: RegressionModel
    Xorig::Vector{T} # original grid points, but sorted
    Yorig::Vector{T} # original values, sorted according to x
    Xrank::Vector{Int}
    Xdesign::Vector{T}
    Xcount::Vector{Int} # to how many observations does X correspond?
    weights::Vector{T}
    RpαQtQ::Matrix{T} # in symmetric banded matrix format
    g::Vector{T} # fitted values
    γ::Vector{T} # 2nd derivatives of fitted vals
    λ::T
end

function fit{T<:LAPACKFloat}(::Type{SmoothingSpline}, X::Vector{T}, Y::Vector{T}, λ::T, ws=ones(Y))
    Xrank = ordinalrank(X) # maybe speed this up when already sorted
    Xorig = sort(X)
    Yorig = sort(Y)

    Xdesign, Xcount = rle(Xorig)

    RpαQtQ = QtQpR(diff(Xdesign), λ)
    pbtrf!('U', 2, RpαQtQ)

    spl = SmoothingSpline{T}(Xorig, Yorig, Xrank, Xdesign, Xcount, ws, RpαQtQ, zeros(Y), zeros(length(Y)-2), λ)
    fit!(spl)
end

function fit!{T<:LAPACKFloat}(spl::SmoothingSpline{T})

    g = spl.g #store Y in g initially
    running_rle_mean!(g, spl.Yorig, spl.Xcount)
    Y = copy(g)
    n = length(spl.g)
    h = diff(spl.Xdesign)
    λ = spl.λ
    Q = ReinschQ(h)

    RpαQtQ = spl.RpαQtQ
    γ = At_mul_B!(spl.γ, Q, g)
    pbtrs!('U', 2, RpαQtQ, γ)
    A_mul_B!(g, Q, γ)
    broadcast!(*, g, g, λ)
    broadcast!(-,g, Y, g)
    spl
end

function fit!{T<:LAPACKFloat}(spl::SmoothingSpline{T}, Y::Vector{T})
    spl.Y = Y[spl.idx]
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

# TODO: this mean should be weighted for weighted case
function running_rle_mean!{T<:Real}(g::Vector{T}, Y::Vector{T}, rlecount::Vector{Int64})
  length(g) == length(rlecount) ||  throw(DimensionMismatch())
  curridx = 1::Int
  for i=1:length(rlecount)
    g[i] = mean(Y[curridx:(curridx+rlecount[i]-1)])
    curridx += rlecount[i]
  end
  g
end

end # module
