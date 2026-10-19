using BenchmarkTools
using Distributions
using Random
using StatsBase
using DataFrames
using Plots
using StatsPlots
using LinearAlgebra
using JuMP
using Ipopt
using Dates
using CSV
using LoopVectorization
using Roots
using QuadGK

include("../library/RiskStats.jl")
include("../library/simulate.jl")
include("../library/return_calculate.jl")
include("../library/fitted_model.jl")
include("../library/missing_cov.jl")
include("../library/ewCov.jl")

#1 - 10 points
# Explain the difference in thinking between data modeling for risk analysis vs data 
# modeling for forecasting. 

# Data modeling for forecasting is focused on the future, and the goal is to predict future values.
# The model uses the expected future value.  Modeling for risk analysis is focused predicting
# the distribution of future values.  This encompasses not just the expected value, but also the
# variance, skewness, kurtosis, and other moments of the distribution.  The goal is to understand
# the potential outcomes and their likelihoods.

# 2. (20 pts) Using problem2.csv 
# a. Calculate the Mean, Variance, Skewness and Kurtosis of the data (8) 
# b. Given a choice between a normal distribution and a t-distribution, which one would 
# you choose to model the data and why based on part a alone (4)? 
# c. Fit both distributions and prove or disprove your choice in b. (8)

#Solution
data = CSV.read("problem2.csv",DataFrame)[!,:X]
#a
mean_data = mean(data)
var_data = var(data)
skew_data = skewness(data)
kurt_data = kurtosis(data)
println("Mean: ", mean_data)
println("Variance: ", var_data)
println("Skewness: ", skew_data)
println("Kurtosis: ", kurt_data)
# Mean: -0.0003457749550813159
# Variance: 0.00048542322361448215
# Skewness: 0.11408120646692742
# Kurtosis: 0.9575595460250139

#b Given the Kurtosis is >0, I would choose the t-distribution to model the data.

#c 
normalFit = fit_normal(data)
tFit = fit_general_t(data)
n_aicc = aicc(normalFit, data)
t_aicc = aicc(tFit, data)
println("Normal AICC: ", n_aicc)
println("T AICC: ", t_aicc)
if n_aicc < t_aicc
    println("Normal Distribution is better")
else
    println("T-Distribution is better")
end
# Normal AICC: -4789.600319016606
# T AICC: -4809.558173851321
# T-Distribution is better

# 3. (20 pts) Using problem2.csv and your fitted models. These are returns of a stock 
# a. Calculate the VaR (5% alpha) as distance from 0 for both models (8) 
# b. Calculate the ES (5% alpha) as the distance from 0 for both models (8) 
# c. Discuss the results. What do you notice? Why is that? (4) 

normalVaR = VaR(normalFit.errorModel)
tVaR = VaR(tFit.errorModel)
println("VaR Normal: $normalVaR, T VaR: $tVaR")

# VaR Normal: 0.036585720113925815, T VaR: 0.03610245171424064

normalES = ES(normalFit.errorModel)
tES = ES(tFit.errorModel)
println("ES Normal: $normalES, T ES: $tES")

# ES Normal: 0.04579212823081352, T ES: 0.04823020079076017

# The VaR values are similar between the two models, with the normal distribution having a slightly higher VaR.
# However, the ES values show a more pronounced difference, with the t-distribution yielding a higher ES.
# This makes sense.  The t-distribution has more kurtosis (heavier tails) than the normal distribution, which means that
# it predicts more extreme losses in the tail of the distribution.  Therefore, while the 95% tile is lower for the t-distribution
# , the ES values reflect the increased risk of extreme losses captured by the t-distribution.

# 4. (10 pts) Using problem4.csv 
# a. Calculate the exponentially weighted correlation matrix with lambda = 0.94 (3) 
# b. Calculate the exponentially weighted variances with lambda=0.97 (3) 
# c. Combine A and B to form a covariance matrix (3) 
# d. Why would you do something like this in practice? (1) 

data4 = Matrix(CSV.read("problem4.csv",DataFrame))
corr_matrix = ewCovar(data4, 0.94)
istd  = diagm(1 ./ sqrt.(diag(corr_matrix)))
corr_matrix = istd * corr_matrix * istd
println("Correlation Matrix: ")
# Correlation Matrix: 
# 3×3 Matrix{Float64}:
#  1.0       0.711329  0.807175
#  0.711329  1.0       0.71302
#  0.807175  0.71302   1.0

vars = (diag(ewCovar(data4, 0.97)))
# Variances:
# 3-element Vector{Float64}:
#  0.015378811498568219
#  0.03551743200920777
#  0.02781346161629677

stds = sqrt.(vars)
covMat = stds' .* corr_matrix .* stds
# Covariance Matrix
# 3×3 Matrix{Float64}:
#  0.0153788  0.0166247  0.0166938
#  0.0166247  0.0355174  0.0224104
#  0.0166938  0.0224104  0.0278135

# In practice we would mix weightings when we think the updating of correlations 
# and volatilies are different.  Here we think the correlations update faster (lower lambda)
# than the variances (higher lambda).  The higher the lambda the more persistance 
# of older data.  The lower the lambda the more weight is given to more recent data.

# 5. (30 pts) Using the data in problem5.csv. These data contain missing values 
# a. Calculate the pairwise covariance of the data (10) 
# b. Is your matrix Positive Definite, Positive Semi-definite, or Non Definite? (5) 
# c. If the matrix is non definite, use Higham’s method to fix the matrix. (10) 
# d. For each principal component, list the variance explained and the cumulative 
# variance explained, sorted from largest to smallest variance explained. (5)

#a.
data=CSV.read("problem5.csv",DataFrame) |> Matrix
c = missing_cov(x; skipMiss=false, fun=cov)
# 5×5 Matrix{Float64}:
#  1.47048   1.45421   0.877269  1.90323  1.44436
#  1.45421   1.25208   0.539548  1.62192  1.23788
#  0.877269  0.539548  1.27242   1.17196  1.09191
#  1.90323   1.62192   1.17196   1.81447  1.58973
#  1.44436   1.23788   1.09191   1.58973  1.39619

#b
c2 = cov2cor(c)
println(min(eigvals(c2)...))
# -0.09482978874911373
# The matrix is non-definite as the smallest eigenvalue is negative.

#c
ch = higham_nearestPSD(c)
# 5×5 Matrix{Float64}:
#  1.47048   1.33236   0.884378  1.6276   1.39956
#  1.33236   1.25208   0.619028  1.4506   1.21445
#  0.884378  0.619028  1.27242   1.07685  1.05966
#  1.6276    1.4506    1.07685   1.81447  1.57793
#  1.39956   1.21445   1.05966   1.57793  1.39619

#d
ev = eigvals(ch)[5:-1:1]
vexp = ev ./ sum(ev)
csexp = round.(cumsum(vexp) * 1000) ./ 1000
DataFrame(:PC => [i for i in 1:5], :Explained => vexp, :Cumulative=>csexp)

# 5×3 DataFrame
#  Row │ PC     Explained     Cumulative 
#      │ Int64  Float64       Float64    
# ─────┼─────────────────────────────────
#    1 │     1   0.897766          0.898
#    2 │     2   0.102234          1.0
#    3 │     3  -8.5402e-11        1.0
#    4 │     4  -1.33757e-10       1.0
#    5 │     5  -1.83898e-10       1.0

# 6. (20 pts) Using problem6.csv.  These data are prices 3 stocks.  You own 100 shares of each stock. 
# Using arithmetic returns:
# a.	De-mean the return series so that the mean of each is 0.  Fit a Student T model for each stock.  
#       Report the fit values. (3)
# b.	Simulate the system using a Gaussian Copula.  Report the correlation matrix you used in the copula. (3)
# c.	What is the VaR and ES at the 5% alpha level for each stock expressed in $? (7)
# d.	What is the VaR and ES at the 5% alpha level for the total portfolio expressed in $? (7)


prices = CSV.read("problem6.csv", DataFrame)
returns = return_calculate(prices,dateColumn="Date")


currentPrice = prices[end,:]
stocks = ["x1", "x2", "x3"]
fits = Dict{String,FittedModel}()
for s in stocks
    #demean
    returns[!,s] .-= mean(returns[!,s])
    fits[s] = fit_general_t(returns[!,s])
    println("Fit for $s: ")
    println("\t $(fits[s].errorModel)")
end
# Fit for x1: 
#          LocationScale{Float64, Continuous, TDist{Float64}}(
# μ: -0.0004786380504944945
# σ: 0.012907570424291352
# ρ: TDist{Float64}(ν=4.729877274742257)
# )

# Fit for x2: 
#          LocationScale{Float64, Continuous, TDist{Float64}}(
# μ: -4.3490115399672916e-5
# σ: 0.00905840741812939
# ρ: TDist{Float64}(ν=6.766901585975631)
# )

# Fit for x3: 
#          LocationScale{Float64, Continuous, TDist{Float64}}(
# μ: 7.48935400214162e-5
# σ: 0.017062723439903806
# ρ: TDist{Float64}(ν=39.86426047892718)
# )

U = DataFrame()
for s in stocks
    U[:,s] = fits[s].u 
end
corsp = corspearman(Matrix(U))
# Internal Coviariance Matrix
# 3×3 Matrix{Float64}:
#  1.0       0.446299  0.394197
#  0.446299  1.0       0.511761
#  0.394197  0.511761  1.0

nSim = 1000000
simU = DataFrame(cdf(Normal(), simulateNormal(nSim,corsp)), stocks)
simRet = DataFrame()
for s in stocks
    simRet[!,s] = fits[s].eval(simU[!,s])
end

portfolio = DataFrame(:Stock=>stocks,
    :currentValue=>Array(currentPrice[stocks]) .* 100)
iteration = [i for i in 1:nSim]
values = crossjoin(portfolio, DataFrame(:iteration=>iteration))

nv = size(values,1)
pnl = Vector{Float64}(undef,nv)
simulatedValue = copy(pnl)
Threads.@threads for i in 1:nv
    simulatedValue[i] = values.currentValue[i] * (1 + simRet[values.iteration[i],values.Stock[i]])
    pnl[i] = simulatedValue[i] - values.currentValue[i]
end

values[!,:pnl] = pnl
values[!,:simulatedValue] = simulatedValue

risk = select(aggRisk(values,[:Stock]),[:Stock, :VaR95, :ES95, :VaR95_Pct, :ES95_Pct])

# PCT Values for reference.  Asked for $ values.
# 4×5 DataFrame
#  Row │ Stock   VaR95    ES95     VaR95_Pct  ES95_Pct  
#      │ String  Float64  Float64  Float64    Float64
# ─────┼────────────────────────────────────────────────
#    1 │ x1      222.714  321.864  0.0268126  0.0387493
#    2 │ x2      133.956  183.998  0.0173059  0.0237709
#    3 │ x3      235.331  298.907  0.0286443  0.0363828
#    4 │ Total   477.606  632.105  0.019685   0.0260529

# do it 1000 times to get a CI 
vars = Array{Float64,2}(undef,(1000,4))
ess = Array{Float64,2}(undef,(1000,4))
for i in 1:1000
    nSim = 1000
    simU = DataFrame(cdf(Normal(), simulateNormal(nSim,corsp;seed=i)), stocks)
    simRet = DataFrame()
    for s in stocks
        simRet[!,s] = fits[s].eval(simU[!,s])
    end

    portfolio = DataFrame(:Stock=>stocks,
        :currentValue=>Array(currentPrice[stocks]) .* 100)
    iteration = [i for i in 1:nSim]
    values = crossjoin(portfolio, DataFrame(:iteration=>iteration))

    nv = size(values,1)
    pnl = Vector{Float64}(undef,nv)
    simulatedValue = copy(pnl)
    for i in 1:nv
        simulatedValue[i] = values.currentValue[i] * (1 + simRet[values.iteration[i],values.Stock[i]])
        pnl[i] = simulatedValue[i] - values.currentValue[i]
    end

    values[!,:pnl] = pnl
    values[!,:simulatedValue] = simulatedValue

    _risk = select(aggRisk(values,[:Stock]),[:Stock, :VaR95, :ES95])
    vars[i,:] = _risk.VaR95'
    ess[i,:] = _risk.ES95'
end

VaR_cl025 = [quantile(vars[:,i],.025) for i in 1:4]
VaR_cl975 = [quantile(vars[:,i],.975) for i in 1:4]
ES_cl025 = [quantile(ess[:,i],.025) for i in 1:4]
ES_cl975 = [quantile(ess[:,i],.975) for i in 1:4]

riskCL = hcat(DataFrame(:VaR_CL025=>VaR_cl025,:VaR_CL975=>VaR_cl975, :ES_CL025=>ES_cl025,:ES_CL975=>ES_cl975))

#Student responses should be inside the 95% CL
# 4×4 DataFrame
#  Row │ VaR_CL025  VaR_CL975  ES_CL025  ES_CL975 
#      │ Float64    Float64    Float64   Float64
# ─────┼──────────────────────────────────────────
#    1 │   202.028    246.906   283.99    364.389
#    2 │   121.519    148.029   163.674   203.744
#    3 │   217.198    256.725   276.384   321.952
#    4 │   434.27     521.733   574.285   686.507