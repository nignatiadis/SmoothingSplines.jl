# SmoothingSplines

[![Build Status](https://github.com/nignatiadis/SmoothingSplines.jl/workflows/CI/badge.svg)](https://github.com/nignatiadis/SmoothingSplines.jl/actions)
[![Coverage](https://codecov.io/gh/nignatiadis/SmoothingSplines.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/nignatiadis/SmoothingSplines.jl)


A Julia package for nonparametric regression with Cubic Smoothing Splines. The initial aim is to provide the same functionality as R's `smooth.spline` function and competitive computational performance. The implementation however is completely independent of the R function and based on the algorithm by Reinsch [1], as described in Chapter 2 of [2].

```julia
using SmoothingSplines
using RDatasets
using Gadfly

cars = dataset("datasets","cars")
X = map(Float64,convert(Array,cars[:Speed]))
Y = map(Float64,convert(Array,cars[:Dist]))

spl = fit(SmoothingSpline, X, Y, 250.0) # λ=250.0
Ypred = predict(spl) # fitted vector
plot(layer(x=X, y=Y, Geom.point),
	layer(x=X, y=Ypred, Geom.line, 	Theme(default_color=colorant"red")))

predict(spl, 20.0) #prediction at arbitrary point
```

### TODO

* Better docs
* conversion between regularization parameter λ and degrees of freedom
* automatic selection of λ (LOOCV, GCV)
* subsampling of design grid for higher efficiency


**References**

[1] Reinsch, Christian H. "Smoothing by spline functions." Numerische mathematik 10.3 (1967): 177-183.

[2] Green, Peter J., and Bernard W. Silverman. Nonparametric regression and generalized linear models: a roughness penalty approach. CRC Press, 1993.
