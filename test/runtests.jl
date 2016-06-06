using SmoothingSplines
using Base.Test

#n=50
#X = rand(n) .* 3
#Y = 2 .* X.^2 - X .- randn(n)
#idx = sortperm(X)
#X = X[idx]
#Y = Y[idx]
#h = diff(X)

# first test some of the helper functions
n=5
Y = rand(n)
h = [1.0 for i=1:(n-1)]

λ = 1.0
Q = SmoothingSplines.ReinschQ(h)
R = SmoothingSplines.ReinschR(h)
Qfull = reshape(Float64[x for x in Q], size(Q))
Rfull = reshape(Float64[x for x in R], size(R))
QtYfull = Qfull'*Y
QtY = At_mul_B!(zeros(n-2), Q, Y)

@test QtY ≈ QtYfull

tmpQtQpRfull = Rfull + λ*Qfull'*Qfull
tmpQtQpR =SmoothingSplines. QtQpR(h, λ)
@test vec(tmpQtQpR[3,:]) ≈ diag(tmpQtQpRfull)
@test vec(tmpQtQpR[2,2:end]) ≈ diag(tmpQtQpRfull,1)
@test vec(tmpQtQpR[1, 3:end]) ≈ diag(tmpQtQpRfull,2)

γfull = tmpQtQpRfull\QtYfull
γ = copy(QtY)
SmoothingSplines.pbtrf!('U', 2, tmpQtQpR)
SmoothingSplines.pbtrs!('U', 2, tmpQtQpR, γ)
@test γ ≈ γfull

gfull = Y - λ*Qfull*γfull
g = Y - λ*A_mul_B!(zeros(Y), Q, γ)
@test g ≈ gfull

# now test the full algorithm, and predict functions
srand(1)
n=50
X = rand(n) .* 3
Y = 2 .* X.^2 - X .- randn(n)

spl = fit(SmoothingSpline, X, Y, 1.0)

@test predict(spl,X) ≈ predict(spl)
