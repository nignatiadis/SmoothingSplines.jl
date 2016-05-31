# SmoothingSplines

[![Build Status](https://travis-ci.org/nignatiadis/SmoothingSplines.jl.svg?branch=master)](https://travis-ci.org/nignatiadis/SmoothingSplines.jl)

A Julia package for nonparametric regression with Cubic Smoothing Splines. The initial aim is to provide the same functionality as R's `smooth.spline` function and competitive computational performance. The implementation however is completely independent of the R function and based on the algorithm by Reinsch [1], as described in Chapter 2 of [2].



**References**

[1] Reinsch, Christian H. "Smoothing by spline functions." Numerische mathematik 10.3 (1967): 177-183.

[2] Green, Peter J., and Bernard W. Silverman. Nonparametric regression and generalized linear models: a roughness penalty approach. CRC Press, 1993.
