using SmoothingSplines
using Test

using LinearAlgebra
using Random

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
X = [0.0; cumsum(h)]
λ = 1.0

ws1 = ones(Float64, length(Y))
ws2 = 1.0 .+ rand(n)
ws_dict = Dict("uniform weights"=> ws1, "random weights"=>ws2)
for (k,ws) in ws_dict
  println(k)
  Q = SmoothingSplines.ReinschQ(h)
  R = SmoothingSplines.ReinschR(h)
  Qfull = reshape(Float64[x for x in Q], size(Q))
  Rfull = reshape(Float64[x for x in R], size(R))
  QtYfull = Qfull'*Y
  QtY = mul!(fill(0.0, n-2), transpose(Q), Y)

  @test QtY ≈ QtYfull


  tmpQtQpRfull = Rfull + λ*Qfull'*Matrix(Diagonal(1.0 ./ ws))*Qfull
  tmpQtQpR =SmoothingSplines. QtQpR(h, λ, ws)
  @test vec(tmpQtQpR[3,:]) ≈ diag(tmpQtQpRfull)
  @test vec(tmpQtQpR[2,2:end]) ≈ diag(tmpQtQpRfull,1)
  @test vec(tmpQtQpR[1, 3:end]) ≈ diag(tmpQtQpRfull,2)

  γfull = tmpQtQpRfull\QtYfull
  γ = copy(QtY)
  SmoothingSplines.pbtrf!('U', 2, tmpQtQpR)
  SmoothingSplines.pbtrs!('U', 2, tmpQtQpR, γ)
  @test γ ≈ γfull

  gfull = Y - λ*Matrix(Diagonal(1.0./ws))*Qfull*γfull
  g = Y - λ*mul!(zero(Y), Q, γ)./ws
  @test g ≈ gfull

  fullfit = fit(SmoothingSpline, X, Y, λ, ws)
end
# now test the full algorithm, and predict functions
println("testing predict function")
Random.seed!(1)
n=50
X = rand(n) .* 3
Y = 2 .* X.^2 .- X .- randn(n)

spl = fit(SmoothingSpline, X, Y, 1.0)

@test predict(spl,X) ≈ predict(spl)

println("testing against smooth.spline on R cars dataset")
# real data examples
# cars dataset
# using RDatasets
# cars = dataset("datasets","cars")
# X = map(Float64,convert(Array,cars[:Speed]))
X = [4.0,4.0,7.0,7.0,8.0,9.0,10.0,10.0,10.0,11.0,11.0,12.0,12.0,12.0,
  12.0,13.0,13.0,13.0,13.0,14.0,14.0,14.0,14.0,15.0,15.0,15.0,16.0,
  16.0,17.0,17.0,17.0,18.0,18.0,18.0,18.0,19.0,19.0,19.0,20.0,20.0,
  20.0,20.0,20.0,22.0,23.0,24.0,24.0,24.0,24.0,25.0]
# Y = map(Float64,convert(Array,cars[:Dist]))
Y = [2.0,10.0,4.0,22.0,16.0,10.0,18.0,26.0,34.0,17.0,28.0,14.0,20.0,24.0,
  28.0,26.0,34.0,34.0,46.0,26.0,36.0,60.0,80.0,20.0,26.0,54.0,32.0,40.0,
  32.0,40.0,50.0,42.0,56.0,76.0,84.0,36.0,46.0,68.0,32.0,48.0,52.0,56.0,
  64.0,66.0,54.0,70.0,92.0,93.0,120.0,85.0]
# so that λ for this package is comparable to λ in smooth.spline
X = (X .- minimum(X))/(maximum(X)-minimum(X))

# compare to R.stats:
# attach(cars)
# cars.spl <- smooth.spline(speed, dist)
λ= 0.1112206 # cars.spl$lambda
# Rpred <- predict(cars.spl)$y
Rpred =  [ 1.657809, 11.682913, 15.064409, 18.482339, 21.947078, 25.465623, 29.050518,
    32.707552, 36.423468, 40.195168, 44.054188, 48.025219, 52.115627, 56.323874,
    60.673887, 69.808295, 74.533703, 79.313597, 84.103688]
cars_fit = fit(SmoothingSpline, X, Y, λ)
@test cars_fit.g ≈ Rpred rtol=1e-4
