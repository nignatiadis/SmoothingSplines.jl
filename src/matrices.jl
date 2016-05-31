import Base.LinAlg.LAPACK: chkstride1, chkuplo, BlasInt, liblapack


immutable ReinschQ{T} <: AbstractMatrix{T}
    h::Vector{T}
end

function Base.size(Q::ReinschQ)
    n = length(Q.h)
   (n+1, n-1)
end

function Base.getindex{T}(Q::ReinschQ{T},i::Int,j::Int)
    h = Q.h
    if (i == j)
        1/h[i]
    elseif (i == j+1)
        -1/h[j] - 1/h[j+1]
    elseif (i == j+2)
        1/h[j+1]
    else
        zero(T)
    end
end

function Base.LinAlg.At_mul_B!(out::AbstractVector, Q::ReinschQ, g::AbstractVector)
    n = length(out)
    h = Q.h
    n == size(Q,2) || throw(DimensionMismatch())
    length(g)   == size(Q,1) || throw(DimensionMismatch())
    Δgp1 = (g[2] - g[1])/h[1]
    @inbounds for i=1:length(out)
        Δg = Δgp1
        Δgp1 = (g[i+2] - g[i+1])/h[i+1]
        out[i] = Δgp1 - Δg
    end
    out
end

function Base.LinAlg.A_mul_B!(out::AbstractVector, Q::ReinschQ, g::AbstractVector)
    n = length(out)
    n == size(Q,1) || throw(DimensionMismatch())
    length(g) == size(Q,2) || throw(DimensionMismatch())
    @inbounds for i = 1:n
        out[i] = 0
        for j = max(1,i-2):min(i,n-2)
            out[i] += g[j]*Q[i,j]
        end
    end
    out
end


immutable ReinschR{T} <: AbstractMatrix{T}
    h::Vector{T}
end

function Base.size(R::ReinschR)
    n = length(R.h)
   (n-1, n-1)
end

function Base.getindex{T}(R::ReinschR{T}, i::Int, j::Int)
    # TODO: add check bounds
    h = R.h
    if (i==j)
        (h[i] + h[i+1])/3
    elseif abs(i-j) == 1
        h[max(i,j)]/6
    else
        zero(T)
    end
end

function QtQpR{T<:Real}(h::Vector{T}, α::T)
    # maybe has some redundant calculations but should be good enough for now
    n = length(h)-1
    Q = ReinschQ(h)
    R = ReinschR(h)
    out = zeros(T, 3, n)
    # start with main diagonal
    for i=1:n
        out[3, i] = α*((Q[i+2,i] - Q[i+1,i])/h[i+1] - (Q[i+1,i] -Q[i, i])/h[i]) + R[i,i]
    end
    # 1st superdiagonal
    for i=1:(n-1)
        # idx (i, i+1)
        out[2,i+1] = α*((Q[i+2, i+1] - Q[i+1, i+1])/h[i+1] - Q[i+1, i+1]/h[i]) + R[i, i+1]
    end
    # 2nd superdiagonal
    for i=1:(n-2)
        # idx (i, i+2)
        out[1, i+2] = α*(Q[i+2, i+2]/h[i+1]) + R[i,i+2]
    end
    out
end

# temporary hack to use LAPACK wrapper,
# as in
# https://github.com/ApproxFun/BandedMatrices.jl/blob/master/src/blas.jl
if VERSION < v"0.5.0-dev"
    macro blasfunc(x)
       return :( $(BLAS.blasfunc(x) ))
    end
else
    import Base.BLAS.@blasfunc
end


for (pbtrf, pbtrs, elty) in
    ((:dpbtrf_,:dpbtrs_,:Float64),
     (:spbtrf_,:spbtrs_,:Float32),
     (:zpbtrf_,:zpbtrs_,:Complex128),
     (:cpbtrf_,:cpbtrs_,:Complex64))
    @eval begin
        # SUBROUTINE DPBTRF( UPLO, N, KD, AB, LDAB, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          UPLO
        #       INTEGER            INFO, KD, LDAB, N
        # *     ..
        # *     .. Array Arguments ..
        #       DOUBLE PRECISION   AB( LDAB, * )
        function pbtrf!(uplo::Char, kd::Integer, AB::StridedMatrix{$elty})
            chkuplo(uplo)
            chkstride1(AB)
            n    = size(AB, 2)
            info = Ref{BlasInt}()
            ccall((@blasfunc($pbtrf), liblapack), Void,
                  (Ptr{UInt8}, Ptr{BlasInt}, Ptr{BlasInt},
                   Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}),
                   &uplo, &n, &kd, AB, &max(1,stride(AB,2)), info)
            AB
        end

        #SUBROUTINE DPBTRS( UPLO, N, KD, NRHS, AB, LDAB, B, LDB, INFO )
        # *     .. Scalar Arguments ..
        #     CHARACTER          UPLO
        #     INTEGER            INFO, KD, LDAB, LDB, N, NRHS
        # *     ..
        # *     .. Array Arguments ..
        #     DOUBLE PRECISION   AB( LDAB, * ), B( LDB, * )
        function pbtrs!(uplo::Char, kd::Integer,
                        AB::StridedMatrix{$elty}, B::StridedVecOrMat{$elty})
            chkuplo(uplo)
            chkstride1(AB, B)
            info = Ref{BlasInt}()
            n    = size(AB,2)
            if n != size(B,1)
                throw(DimensionMismatch("Matrix AB has dimensions $(size(AB)), but right hand side matrix B has dimensions $(size(B))"))
            end
            ccall((@blasfunc($pbtrs), liblapack), Void,
                  (Ptr{UInt8}, Ptr{BlasInt}, Ptr{BlasInt},  Ptr{BlasInt},
                   Ptr{$elty}, Ptr{BlasInt}, Ptr{$elty},   Ptr{BlasInt},
                   Ptr{BlasInt}),
                  &uplo, &n, &kd, &size(B,2), AB,  &max(1,stride(AB,2)),
                  B, &max(1,stride(B,2)), info)
            B
        end
    end
end
