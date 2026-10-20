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
using MarketData

include("../library/RiskStats.jl")
include("../library/simulate.jl")
include("../library/return_calculate.jl")
include("../library/fitted_model.jl")
include("../library/missing_cov.jl")
include("../library/ewCov.jl")

#1 - 20 points
#using problem1.csv.  
#a. (2)calcaulte mean, variance, skewness, kurtosis of the data
#b. (3) given a choice between a normal distribution and a t-distribution, 
# which one would you choose to model the data? Why?
#c. (4) fit both disitributions and prove or disprove your choice in b.
#d. (4) calculate the 5% and 1% VaR and ES for each distribution.  
#e. (3) calculate the 5% and 1% VaR and ES for the data using historical simulation.
#f. (4) compare the results in d and e.  How does this line up with your choice in b and results in c?


#Data Generation:
Random.seed!(90210)
d = TDist(25)*.03 
data = rand(d, 1000)
CSV.write("problem1.csv", DataFrame(:X=>data))

data = CSV.read("problem1.csv", DataFrame).X
mean_data = mean(data)
std_data = std(data)
skew_data = skewness(data)
kurt_data = kurtosis(data)
println("Mean: $mean_data, \nStd Dev: $std_data, \nSkewness: $skew_data, \nKurtosis: $kurt_data")
#ANSWERS:
# Mean: -0.0005316702326786278, 
# Std Dev: 0.03172783789203951,
# Skewness: -0.2395102766006735,
# Kurtosis: 0.680438623588139

# Kurosis is low but not 0.  This is probably somewhere in the middle of the two distributions.  
# I would lean T here but would not be surprised if the Normal fits better on an AICC test.

nf = fit_normal(data)
tf = fit_general_t(data)
println("Normal Fit: mean=$(nf.errorModel.μ), std=$(nf.errorModel.σ)")
println("T Fit: mean=$(tf.errorModel.μ), s=$(tf.errorModel.σ), ν=$(tf.errorModel.ρ.ν)")

# Normal Fit: mean=-0.0005316702326786278, std=0.03172783789203951
# T Fit: mean=-0.00021915305035255959, s=0.029299141819502817, ν=13.711855137418015
naicc = aicc(nf,data)
taicc = aicc(tf,data)
println("Normal AICC: $naicc, T AICC: $taicc")
# Normal AICC: -4060.2325305235076, T AICC: -4068.109480219482
# T AICC is less than Normal AICC, so T is a better fit.

println("Normal VaR 5%: $(VaR(nf.errorModel, alpha=0.05)), ES 5%: $(ES(nf.errorModel, alpha=.05))")
println("Normal VaR 1%: $(VaR(nf.errorModel, alpha=0.01)), ES 1%: $(ES(nf.errorModel, alpha=.01))")
println("T VaR 5%: $(VaR(tf.errorModel, alpha=0.05)), ES 5%: $(ES(tf.errorModel, alpha=.05))")
println("T VaR 1%: $(VaR(tf.errorModel, alpha=0.01)), ES 1%: $(ES(tf.errorModel, alpha=.01))")

# Normal VaR 5%: 0.05271931946472819, ES 5%: 0.06597708780254659
# Normal VaR 1%: 0.07434165846073733, ES 1%: 0.08509315494056131
# T VaR 5%: 0.05190076384436333, ES 5%: 0.06771378728187447
# T VaR 1%: 0.07731990069879922, ES 1%: 0.092325522076692

println("Historical VaR 5%: $(VaR(data, alpha=0.05)), ES 5%: $(ES(data, alpha=0.05))")
println("Historical VaR 1%: $(VaR(data, alpha=0.01)), ES 1%: $(ES(data, alpha=0.01))")

# Historical VaR 5%: 0.05365484666259305, ES 5%: 0.07107272387024315
# Historical VaR 1%: 0.07737344675790993, ES 1%: 0.09905041150582906
# On a Historical VaR basis, neither is a perfect fit.  The Normal VaR
# underestimates the risk at both levels, while the T VaR underestimates 
# for all except the ES 1% value. The T distribution has roughly the 
# same problem.  This is unsurprising given the closeness of the kurtosis to 0
# neither is account for the negative skewness which is why they are 
# underestimating the risk.  The T distribution is a better fit, but neither is perfect.


#2 - 20 points
# using problem2.csv.

# You and your team have done research into the speed at which correlations and
# variances change through time.  You have found that the speed of change is 
# different for correlations vs variances.  Correlations are slower moving
# but variances update faster.

#a. (5) Given that you have decided to use an exponentially weighted correlation
# and variance estimator, and will use a different lambda for each.  Should
# you choose a higher or lower lambda for the correlation estimator?  Why?
#b. (5) Given your choice in a, and possible λ values of 0.94 and 0.97, calculate
# the exponentially weighted correlation for the input data.
#c. (5) Given your choice in a, and possible λ values of 0.94 and 0.97, calculate
# the exponentially weighted variance for the input data.
#d. (5) Combine the results in a final covariance matrix.  

#data Generation:
Random.seed!(8675309)
n = 1000
p = 5   
_corr = Array{Float64}(undef, p, p)
for i in 1:p
    for j in i:p
        if i == j
            _corr[i, j] = 1.0
        else
            _corr[i, j] = rand(Uniform(-0.5, 0.5))
            _corr[j, i] = _corr[i, j]
        end
    end
end
_stds = rand(Uniform(0.01, 0.05), p)
cov_matrix = Diagonal(_stds) * _corr * Diagonal(_stds)
cov_matrix = (cov_matrix + cov_matrix') ./ 2
data = rand(MvNormal(zeros(p), cov_matrix), n)'
CSV.write("problem2.csv", DataFrame(data, :auto))

#a. You would choose the higher lambda for the correlation estimator as higher values
# have slower decay rates.  This means we have more equal weighting across the data 
# and less bias towards recent values.

data = Matrix(CSV.read("problem2.csv", DataFrame))

_fit_corr = ewCovar(data, 0.97)
_fit_corr = _fit_corr ./ (sqrt.(diag(_fit_corr)) * sqrt.(diag(_fit_corr))')
println("Exponentially Weighted Correlation Matrix (λ=0.97):")
display(_fit_corr)
#b.
# Exponentially Weighted Correlation Matrix (λ=0.97):
# 5×5 Matrix{Float64}:
#   1.0        0.501728  -0.276033   -0.153536   -0.287344
#   0.501728   1.0        0.312724   -0.321841   -0.222609
#  -0.276033   0.312724   1.0        -0.0731389   0.418567
#  -0.153536  -0.321841  -0.0731389   1.0        -0.406487
#  -0.287344  -0.222609   0.418567   -0.406487    1.0

_fit_cov = ewCovar(data, 0.94)
sd = Diagonal(sqrt.(diag(_fit_cov)))
_fit_cov = sd * _fit_corr * sd
#c 
println("Exponentially Weighted Standard Deviation Matrix (λ=0.94):")
display(sd)
println("As Variance")
display(sd .* sd)
# Exponentially Weighted Standard Deviation Matrix (λ=0.94):
# 5×5 Diagonal{Float64, Vector{Float64}}:
#  0.0187983   ⋅          ⋅          ⋅          ⋅ 
#   ⋅         0.0235618   ⋅          ⋅          ⋅
#   ⋅          ⋅         0.0136579   ⋅          ⋅
#   ⋅          ⋅          ⋅         0.0204931   ⋅
#   ⋅          ⋅          ⋅          ⋅         0.0112255
# As Variance
# 5×5 Diagonal{Float64, Vector{Float64}}:
#  0.000353376   ⋅            ⋅            ⋅            ⋅
#   ⋅           0.000555158   ⋅            ⋅            ⋅
#   ⋅            ⋅           0.000186538   ⋅            ⋅
#   ⋅            ⋅            ⋅           0.000419965   ⋅
#   ⋅            ⋅            ⋅            ⋅           0.000126011

#d. The final covariance matrix is:
println("Exponentially Weighted Modeled Covariance:")
display(_fit_cov)
# Exponentially Weighted Modeled Covariance:
# 5×5 Matrix{Float64}:
#   0.000353376   0.000222226  -7.08703e-5   -5.91472e-5   -6.06351e-5
#   0.000222226   0.000555158   0.000100636  -0.000155402  -5.88782e-5
#  -7.08703e-5    0.000100636   0.000186538  -2.0471e-5     6.41731e-5
#  -5.91472e-5   -0.000155402  -2.0471e-5     0.000419965  -9.351e-5
#  -6.06351e-5   -5.88782e-5    6.41731e-5   -9.351e-5      0.000126011

#3 - 20 points
# using problem3.csv.   
# You are given the input covariance matrix and need to use it for 
# risk analysis.
#a. (5) Calculate the eigenvalues of the covariance matrix.  What do you see?
#b. (5) Calculate the nearest PSD matrix using Higham's algorithm.  How does this change the eigenvalues?
#c. (5) Calculate the nearest PSD matrix using the Near PSD method of Rebonato and Jäckel.  How does this change the eigenvalues?
#d. (5) Compare the results in b and c.  Which method do you prefer and why?

#data Generation:
Random.seed!(420)
p = 5
_corr = Array{Float64}(undef, p, p)
for i in 1:p
    for j in i:p
        if i == j
            _corr[i, j] = 1.0
        else
            _corr[i, j] = rand(Uniform(-0.5, 0.5))
            _corr[j, i] = _corr[i, j]
        end
    end
end
_corr[1, 2] = .9324
_corr[2, 1] = .9324
sd = rand(Uniform(0.01, 0.05), p)
_cov = Diagonal(sd) * _corr * Diagonal(sd)
_cov = (_cov + _cov') ./ 2
CSV.write("problem3.csv", DataFrame(_cov, :auto))

data = Matrix(CSV.read("problem3.csv", DataFrame))
eigenvalues = eigvals(data)
println("Eigenvalues of the covariance matrix:")
display(eigenvalues)
# a.
# Eigenvalues of the covariance matrix:
# 5-element Vector{Float64}:
#  -4.5616001351745625e-5
#   0.00012356677195028497
#   0.0001908296373330973
#   0.0004267281595771177
#   0.0008649156097302901
# There is a significant negative eigenvalue, which indicates that the matrix is not positive semi-definite.  
# This could lead to issues in risk analysis as it may produce negative variances or other non-sensical results.

#b. 
higham = higham_nearestPSD(data)
higham_eigenvalues = eigvals(higham)
println("Higham Adjusted Matrix")
display(higham)
println("Eigenvalues of the Higham nearest PSD matrix:")    
display(higham_eigenvalues)
# Higham Adjusted Matrix
# 5×5 Matrix{Float64}:
#   0.000337588   0.000345672  -6.7811e-5    -4.28635e-5   -3.62003e-5
#   0.000345672   0.000501453   2.01492e-6   -0.00016224    5.46521e-5
#  -6.7811e-5     2.01492e-6    0.000189133  -0.000101158  -4.63366e-5
#  -4.28635e-5   -0.00016224   -0.000101158   0.000403089  -2.92798e-5
#  -3.62003e-5    5.46521e-5   -4.63366e-5   -2.92798e-5    0.000129162
# Eigenvalues of the Higham nearest PSD matrix:
# 5-element Vector{Float64}:
#  -3.03464501740852e-13
#   0.00011853862925737185
#   0.0001877965117750448
#   0.0004218167379770747
#   0.0008322722985330185
# The Higham method removed the negative eigenvalue and set it to 0, 
# while leaving the other eigenvalues relatively unchanged.

# c.
nearPsd = near_psd(data)
nearPsd_eigenvalues = eigvals(nearPsd)
println("Near PSD Adjusted Matrix")
display(nearPsd)
println("Eigenvalues of the Near PSD matrix:")
display(nearPsd_eigenvalues)
# Near PSD Adjusted Matrix
# 5×5 Matrix{Float64}:
#   0.000337588   0.000339766  -6.84556e-5   -4.11224e-5   -3.79551e-5
#   0.000339766   0.000501453   5.53656e-6   -0.000158481   5.64025e-5
#  -6.84556e-5    5.53656e-6    0.000189133  -0.000100357  -4.64565e-5
#  -4.11224e-5   -0.000158481  -0.000100357   0.000403089  -2.88686e-5
#  -3.79551e-5    5.64025e-5   -4.64565e-5   -2.88686e-5    0.000129162
# Eigenvalues of the Near PSD matrix:
# 5-element Vector{Float64}:
#  4.859029786141216e-20
#  0.00012478338451461581
#  0.0001885834913011492
#  0.0004223523275268444
#  0.0008247049738964351
# The Near PSD method also removed the negative eigenvalue and kept the other eigenvalues relatively unchanged,
# though it appears to have a slightly smaller largest eigenvalue than the Higham method.

#d. 
println("Eignvalues Comparison:")
display(DataFrame(hcat(eigenvalues, higham_eigenvalues, nearPsd_eigenvalues),[:Original, :Higham, :NearPSD]))
# Eignvalues Comparison:
# 5×3 DataFrame
#  Row │ Original      Higham        NearPSD     
#      │ Float64       Float64       Float64
# ─────┼─────────────────────────────────────────
#    1 │ -4.5616e-5    -3.03465e-13  4.85903e-20
#    2 │  0.000123567   0.000118539  0.000124783
#    3 │  0.00019083    0.000187797  0.000188583
#    4 │  0.000426728   0.000421817  0.000422352
#    5 │  0.000864916   0.000832272  0.000824705

dhigheig = sum((eigenvalues - higham_eigenvalues).^2)
dneareig = sum((eigenvalues - nearPsd_eigenvalues).^2)
println("Distance of Higham eigenvalues from original: $dhigheig")
println("Distance of Near PSD eigenvalues from original: $dneareig")
# Distance of Higham eigenvalues from original: 3.2050094492832762e-9
# Distance of Near PSD eigenvalues from original: 3.7233880377460915e-9

diff_higham = abs.(higham - data)
diff_nearPsd = abs.(nearPsd - data)
println("Absolute Differences between Higham and Original:")
display(diff_higham)
println("Sum of Absolute Differences for Higham: $(sum(diff_higham))")
println("Absolute Differences between Near PSD and Original:")  
display(diff_nearPsd)
println("Sum of Absolute Differences for Near PSD: $(sum(diff_nearPsd))")
# Absolute Differences between Higham and Original:
# 5×5 Matrix{Float64}:
#  0.0         3.79567e-5  1.02492e-5   3.6014e-7   1.01215e-5
#  3.79567e-5  0.0         1.19194e-5   4.18825e-7  1.17708e-5
#  1.02492e-5  1.19194e-5  5.42101e-20  1.13093e-7  3.17841e-6
#  3.6014e-7   4.18825e-7  1.13093e-7   0.0         1.11684e-7
#  1.01215e-5  1.17708e-5  3.17841e-6   1.11684e-7  0.0
# Sum of Absolute Differences for Higham: 0.00017239954909766157
# Absolute Differences between Near PSD and Original:
# 5×5 Matrix{Float64}:
#  1.0842e-19  4.3863e-5   9.60462e-6   1.38096e-6  8.36673e-6
#  4.3863e-5   0.0         8.39771e-6   4.17804e-6  1.00204e-5
#  9.60462e-6  8.39771e-6  5.42101e-20  6.87845e-7  3.05852e-6
#  1.38096e-6  4.17804e-6  6.87845e-7   0.0         2.99557e-7
#  8.36673e-6  1.00204e-5  3.05852e-6   2.99557e-7  0.0
# Sum of Absolute Differences for Near PSD: 0.00017971478760808653

# Both methods corrected the negative eigenvalue and produced a PSD matrix, 
# but the Higham method resulted in a smaller distance from the original eigenvalues 
# and a smaller sum of absolute differences from the original matrix.  
# Given the small matrix, the time to compute is negligible for both methods, so I would 
# prefer the method that produces a result closer to the original matrix -- Higham.

#4 - 60 points
# Using the price data in problem4.csv
# You hold a portfolio with a current value of $100,000
# a. (5) Calculate the number of shares (fractions of shares are OK) of each of these stocks in your portfolio so that each stock 
# has an equal weight.
# b. (10) Calculate the daily returns of each stock using arthemetic returns.  Show the first 5 rows and last 5 rows of the return data.
# c. (15) Remove the mean from each series.  Fit both a normal and a t-distribution to the returns of each stock.  Show the parameters of best fit for each stock.
# d. (30) Calculate the 1% VaR and ES for each stock and the portfolio.  Use a gaussian copula to model the dependence structure 
# between the stocks.  Calculate the 1% VaR and ES using a historical simulation as well. Present the results in a table and 
# compare the results.  Which method do you prefer and why?
# (5) Individual stock VaR and ES via simulation
# (5) Historic Simulation 
# (10) Portfolio VaR and ES via simulation
# (10) discussion of results

#data Generation
tickers = ["SPY", "AAPL", "MSFT", "GOOGL", "BABA"]

histPrices = DataFrame[]

for t in Symbol.(tickers)
    df = DataFrames.rename(DataFrame(yahoo(t, YahooOpt(period1=DateTime(2022,1,1)))), Dict(:timestamp=>:Date, :AdjClose=>t))[!,[:Date, t]]
    append!(histPrices,[df])
end
prices = innerjoin(histPrices...,on=:Date)
CSV.write("problem4.csv",prices)

# a.
prices = CSV.read("problem4.csv", DataFrame)
tickers = names(prices)[2:end]
current_prices = prices[end, 2:end]
vec_cprices = [ current_prices... ]
portfolio_value = 100000.0
weights = fill(1.0 / length(tickers), length(tickers))  
shares = (weights .* portfolio_value) ./ vec_cprices
portfolio = DataFrame(:Ticker=>tickers, :Shares=>shares)
println("Number of shares for each stock:")
display(portfolio)
# Number of shares for each stock:
# 5×2 DataFrame
#  Row │ Ticker  Shares   
#      │ String  Float64
# ─────┼──────────────────
#    1 │ SPY      29.0095
#    2 │ AAPL     75.5915
#    3 │ MSFT     50.3487
#    4 │ GOOGL    63.4961
#    5 │ BABA    129.492

#b.
returns = return_calculate(prices, dateColumn="Date")
println("First 5 rows of returns:")
display(first(returns, 5))
println("Last 5 rows of returns:")
display(last(returns, 5))
# First 5 rows of returns:
# 5×6 DataFrame
#  Row │ Date        SPY           AAPL          MSFT          GOOGL         BABA        
#      │ Date        Float64       Float64       Float64       Float64       Float64
# ─────┼─────────────────────────────────────────────────────────────────────────────────
#    1 │ 2022-01-04  -0.000335043  -0.0126915    -0.0171469    -0.00408296   -0.00681181
#    2 │ 2022-01-05  -0.019202     -0.0266001    -0.038388     -0.0458762     0.0133824
#    3 │ 2022-01-06  -0.000939602  -0.0166933    -0.00790205   -0.000199743   0.0451469
#    4 │ 2022-01-07  -0.00395347    0.000988372   0.000510058  -0.00530295    0.0251125
#    5 │ 2022-01-10  -0.00124442    0.000116201   0.000732559   0.0120604    -0.0116323
# Last 5 rows of returns:
# 5×6 DataFrame
#  Row │ Date        SPY           AAPL         MSFT         GOOGL        BABA        
#      │ Date        Float64       Float64      Float64      Float64      Float64
# ─────┼──────────────────────────────────────────────────────────────────────────────
#    1 │ 2026-02-13   0.000704538  -0.0227334   -0.00129402  -0.0106149   -0.0189
#    2 │ 2026-02-17   0.00161346    0.0316679   -0.0111134   -0.0121026   -0.00192643
#    3 │ 2026-02-18   0.00503771    0.00178112   0.00690425   0.00433745   0.00218755
#    4 │ 2026-02-19  -0.00263737   -0.0142615   -0.00285289  -0.00158237  -0.00962958
#    5 │ 2026-02-20   0.00723179    0.0153504   -0.00308684   0.0400528    0.00116674

#c.
for t in tickers
    returns[!, Symbol(t)] .-= mean(returns[!, Symbol(t)])
end

fittedModels = Dict{String,FittedModel}()
for t in tickers
    ft = fit_general_t(returns[!, Symbol(t)])
    fn = fit_normal(returns[!, Symbol(t)])
    if aicc(ft, returns[!, Symbol(t)]) < aicc(fn, returns[!, Symbol(t)])
        fittedModels[t] = ft
    else
        fittedModels[t] = fn
    end
    println("Fitted model for $t: $(typeof(fittedModels[t].errorModel)) with parameters: $(fittedModels[t].errorModel)")
end
# Fitted model for SPY: LocationScale{Float64, Continuous, TDist{Float64}} with parameters: LocationScale{Float64, Continuous, TDist{Float64}}(
# μ: 0.00027858287499741804
# σ: 0.007722799294132536
# ρ: TDist{Float64}(ν=3.6648954473348256)
# )
# Fitted model for AAPL: LocationScale{Float64, Continuous, TDist{Float64}} with parameters: LocationScale{Float64, Continuous, TDist{Float64}}(
# μ: 0.00016870427558700871
# σ: 0.012190250060417279
# ρ: TDist{Float64}(ν=3.492939841648696)
# )
# Fitted model for MSFT: LocationScale{Float64, Continuous, TDist{Float64}} with parameters: LocationScale{Float64, Continuous, TDist{Float64}}(
# μ: 0.0002061935633905086
# σ: 0.012529025663853928
# ρ: TDist{Float64}(ν=4.126273032944454)
# )
# Fitted model for GOOGL: LocationScale{Float64, Continuous, TDist{Float64}} with parameters: LocationScale{Float64, Continuous, TDist{Float64}}(
# μ: 3.521150098278354e-5
# σ: 0.015416426707686602
# ρ: TDist{Float64}(ν=4.5520248353202035)
# )
# Fitted model for BABA: LocationScale{Float64, Continuous, TDist{Float64}} with parameters: LocationScale{Float64, Continuous, TDist{Float64}}(
# μ: -0.0021603923076874035
# σ: 0.020769639158760393
# ρ: TDist{Float64}(ν=3.1251949227351306)
# )

#d.
Random.seed!(31415)
nTimes = 1000
VaRs = DataFrame(fill(0.0, (nTimes, length(tickers)+1)), [Symbol.(tickers)...,:Total]   )
ESs = DataFrame(fill(0.0, (nTimes, length(tickers)+1)), [Symbol.(tickers)...,:Total]   )

# I'm running a 1000 times to get a distribution of possible values
for _t in 1:nTimes
    U = DataFrame()
    for t in tickers
        U[!, Symbol(t)] = fittedModels[t].u
    end
    spcorr = corspearman(Matrix(U))
    NSim = 1000
    simU = DataFrame(
                #convert standard normals to U
                cdf(Normal(),
                    simulate_pca(spcorr,NSim;seed=_t)  #simulation the standard normals
                )   
                , tickers
            )
    simulatedReturns = DataFrame()
    for t in tickers
        simulatedReturns[!, Symbol(t)] = fittedModels[t].eval(simU[!, Symbol(t)])
    end

    iteration = [i for i in 1:NSim]
    values = crossjoin(portfolio, DataFrame(:iteration=>iteration))

    nVals = size(values,1)
    currentValue = Vector{Float64}(undef,nVals)
    simulatedValue = Vector{Float64}(undef,nVals)
    pnl = Vector{Float64}(undef,nVals)
    for i in 1:nVals
        price = current_prices[values.Ticker[i]]
        currentValue[i] = values.Shares[i] * price
        simulatedValue[i] = values.Shares[i] * price*(1.0+simulatedReturns[values.iteration[i],values.Ticker[i]])
        pnl[i] = simulatedValue[i] - currentValue[i]
    end
    values[!,:currentValue] = currentValue
    values[!,:simulatedValue] = simulatedValue
    values[!,:pnl] = pnl

    risk = aggRisk(values,[:Ticker])
    for n in names(VaRs)
        VaRs[_t, n] = risk[risk.Ticker .== n, :VaR99][1]
        ESs[_t, n] = risk[risk.Ticker .== n, :ES99][1]
    end
end

simVaR  = DataFrame(hcat(fill("",3),fill(0.0, (3, length(tickers)+1))), [:Metric, Symbol.(tickers)...,:Total]   )
simES  = DataFrame(hcat(fill("",3),fill(0.0, (3, length(tickers)+1))), [:Metric, Symbol.(tickers)...,:Total]   )
for n in [Symbol.(tickers)...,:Total]
    simVaR[!, :Metric] = ["Mean", "Upper95CI", "Lower95CI"]
    simVaR[1, n] = mean(VaRs[!, n])
    simVaR[2, n] = quantile(VaRs[!, n], 0.975) 
    simVaR[3, n] = quantile(VaRs[!, n], 0.025)
    simES[1, n] = mean(ESs[!, n])
    simES[2, n] = quantile(ESs[!, n], 0.975)
    simES[3, n] = quantile(ESs[!, n], 0.025)
end

println("Simulated VaR with Gaussian Copula:")
display(simVaR)
println("Simulated ES with Gaussian Copula:")
display(simES)
# Student responses should be within the 95% confidence intervals of these values, 
# but may not be exactly the same due to the randomness in the simulation.
# Simulated VaR with Gaussian Copula:
# 3×7 DataFrame
#  Row │ Metric     SPY      AAPL     MSFT     GOOGL    BABA     Total   
#      │ String     Any      Any      Any      Any      Any      Any
# ─────┼─────────────────────────────────────────────────────────────────
#    1 │ Mean       615.004  1010.09  940.507  1095.89  1908.41  4031.84
#    2 │ Upper95CI  768.556  1270.84  1141.26  1316.9   2371.14  4802.64
#    3 │ Lower95CI  505.999  820.611  777.128  920.627  1520.34  3395.64
# Simulated ES with Gaussian Copula:
# 3×7 DataFrame
#  Row │ Metric  SPY      AAPL     MSFT     GOOGL    BABA     Total   
#      │ Any     Any      Any      Any      Any      Any      Any
# ─────┼──────────────────────────────────────────────────────────────
#    1 │         854.407  1416.23  1264.16  1438.95  2761.72  5264.87
#    2 │         1193.65  2026.89  1670.82  1888.71  4135.78  6974.91
#    3 │         637.494  1055.54  980.742  1140.7   1980.25  4144.49

#Historical Simulation
NSim = size(returns, 1)

iteration = [i for i in 1:NSim]
values = crossjoin(portfolio, DataFrame(:iteration=>iteration))

nVals = size(values,1)
currentValue = Vector{Float64}(undef,nVals)
simulatedValue = Vector{Float64}(undef,nVals)
pnl = Vector{Float64}(undef,nVals)
for i in 1:nVals
    price = current_prices[values.Ticker[i]]
    currentValue[i] = values.Shares[i] * price
    simulatedValue[i] = values.Shares[i] * price*(1.0+returns[values.iteration[i],values.Ticker[i]])
    pnl[i] = simulatedValue[i] - currentValue[i]
end
values[!,:currentValue] = currentValue
values[!,:simulatedValue] = simulatedValue
values[!,:pnl] = pnl
risk = aggRisk(values,[:Ticker])[!,[:Ticker, :VaR99, :ES99]]
println("Historical Simulation VaR and ES:")
display(risk)
# Historical Simulation VaR and ES:
# 6×3 DataFrame
#  Row │ Ticker  VaR99     ES99     
#      │ String  Float64   Float64
# ─────┼────────────────────────────
#    1 │ SPY      660.546   835.112
#    2 │ AAPL     977.308  1181.64
#    3 │ MSFT     879.863  1184.15
#    4 │ GOOGL   1018.1    1426.58
#    5 │ BABA    1673.57   2024.36
#    6 │ Total   3826.89   4700.98

DataFrames.rename!(risk, :VaR99=>:HistoricalVaR99, :ES99=>:HistoricalES99)
risk[!, :SimulatedVaR99] = [simVaR[1, 2:end]...]
risk[!, :SimulatedES99] = [simES[1, 2:end]...]
risk = risk[:, [:Ticker, :HistoricalVaR99, :SimulatedVaR99, :HistoricalES99, :SimulatedES99]]
println("Comparison of Historical Simulation and Simulated VaR and ES:")    
display(risk)
# Comparison of Historical Simulation and Simulated VaR and ES:
# 6×5 DataFrame
#  Row │ Ticker  HistoricalVaR99  SimulatedVaR99  HistoricalES99  SimulatedES99 
#      │ String  Float64          Float64         Float64         Float64
# ─────┼────────────────────────────────────────────────────────────────────────
#    1 │ SPY             660.546         615.004         835.112        854.407
#    2 │ AAPL            977.308        1010.09         1181.64        1416.23
#    3 │ MSFT            879.863         940.507        1184.15        1264.16
#    4 │ GOOGL          1018.1          1095.89         1426.58        1438.95
#    5 │ BABA           1673.57         1908.41         2024.36        2761.72
#    6 │ Total          3826.89         4031.84         4700.98        5264.87

# NOTE: Any answer by the student that is a thoughtful analysis of the differences between the 
# historical simulation and the simulated method should be accepted.

# Our simulated risk values seem to generally overestimate the risk compared to the historical simulation. 
# We are looking at the 1% risk level and with 1000 samples in the history, that means we are
# looking at nearly the worst value in history, but not the absolute worst.  
# Our simulated values do not take into account any skewnes
# in the input data, which could also be leading to the higher values.

println("BABA Historic Skew: $(skewness(returns.BABA))")
# BABA Historic Skew: 1.7293276358094927
# For example BABA is positively skewed in the historical data.  That would explain why our simulated 
# values are higher than the historical values, as we are not accounting for the skewness in the data.

# Given the large sample size and our inability to account for skewness in the simulated method, 
# I would prefer the historical simulation results as they are more likely to reflect the true 
# risk of the portfolio.
