module SmoothingSplines

import StatsBase: fit!, fit, RegressionModel

export SmoothingSpline, fit!, fit


LAPACKFloat = Union{Float32,Float64}

include("matrices.jl")

type SmoothingSpline{T<:LAPACKFloat} <: RegressionModel
    X::Vector{T}
    Y::Vector{T} # eventually use GLM response objects?
    perm::Vector{Int}
    h::Vector{T}
    RpαQtQ::Matrix{T} # in symmetric banded matrix format
    g::Vector{T} # fitted values
    γ::Vector{T} # 2nd derivatives of fitted vals
    λ::T
end

function fit{T<:LAPACKFloat}(::Type{SmoothingSpline}, X::Vector{T}, Y::Vector{T}, λ::T)
    idx = sortperm(X)
    X = X[idx]
    n = length(X)
    Y = Y[idx]
    h = diff(X)

    RpαQtQ = QtQpR(h, λ)
    pbtrf!('U', 2, RpαQtQ)

    spl = SmoothingSpline{T}(X, Y, idx, h, RpαQtQ, zeros(Y), zeros(length(Y)-2), λ)
    fit!(spl)

end

function fit!{T<:LAPACKFloat}(spl::SmoothingSpline{T})
    Y = spl.Y
    g = spl.g
    n = length(spl.Y)
    h = spl.h
    λ = spl.λ
    Q = ReinschQ(h)

    RpαQtQ = spl.RpαQtQ
    QtY = At_mul_B!(spl.γ, Q, Y)
    pbtrs!('U', 2, RpαQtQ, QtY)
    A_mul_B!(g, Q, QtY)
    broadcast!(*, g, g, λ)
    broadcast!(-,g, Y, g)
    spl
end

function fit!{T<:LAPACKFloat}(spl::SmoothingSpline{T}, Y::Vector{T})
    spl.Y = Y[spl.idx]
    fit!(spl)
end

end # module
