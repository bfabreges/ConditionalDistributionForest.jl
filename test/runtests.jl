using ConditionalDistributionForest
using DecisionTree

using Test
using Random
using Distributions

@testset "Building forest" begin
    rng = MersenneTwister(1234)
    
    n_samples = 10
    X = Array{Float64, 2}(undef, n_samples, 3)
    X[:, 1] .= rand(rng, GeneralizedPareto(0., 1.5, 0.25), n_samples)
    X[:, 2] .= rand(rng, LogNormal(1.1, 0.6), n_samples)
    X[:, 3] .= rand(rng, Gamma(2, 5.0/3), n_samples)
    
    Y = vec(sum(X, dims=2)) + rand(rng, Normal(0, 2), n_samples)

    rng = MersenneTwister(5678)
    forest_1, bootstraps = build_distribution_forest(Y, X, -1, 1, 1.0, -1, 1, rng = rng, return_bootstrap = true)
    
    rng = MersenneTwister(5678)
    forest_2 = build_distribution_forest(Y, X, -1, 1, 1.0, -1, 1, rng = rng)

    res = Array{Int64, 2}(undef, n_samples, 1)
    res[:, 1] = [1; 3; 5; 5; 8; 3; 1; 9; 4; 10]
    
    original_stdout = stdout
    (rd, wr) = redirect_stdout()
    DecisionTree.print_tree(forest_1.trees[1])
    redirect_stdout(original_stdout)
    close(wr)
    data_1 = read(rd, String)
    close(rd)
    
    (rd, wr) = redirect_stdout()
    DecisionTree.print_tree(forest_2.trees[1])
    redirect_stdout(original_stdout)
    close(wr)
    data_2 = read(rd, String)
    close(rd)

    @test bootstraps == res
    @test data_1 == data_2
end


@testset "Computing forest weights" begin
    rng = MersenneTwister(1234)

    n_primes = 2
    Xvec = Array{Float64, 2}(undef, n_primes, 3)
    Xvec[:, 1] .= rand(rng, GeneralizedPareto(0., 1.5, 0.25), n_primes)
    Xvec[:, 2] .= rand(rng, LogNormal(1.1, 0.6), n_primes)
    Xvec[:, 3] .= rand(rng, Gamma(2, 5.0/3), n_primes)
    
    n_samples = 10
    X = Array{Float64, 2}(undef, n_samples, 3)
    X[:, 1] .= rand(rng, GeneralizedPareto(0., 1.5, 0.25), n_samples)
    X[:, 2] .= rand(rng, LogNormal(1.1, 0.6), n_samples)
    X[:, 3] .= rand(rng, Gamma(2, 5.0/3), n_samples)
    
    Y = vec(sum(X, dims=2)) + rand(rng, Normal(0, 2), n_samples)

    forest, bootstraps = build_distribution_forest(Y, X, -1, 1, 1.0, -1, 1, rng = rng, return_bootstrap = true)

    @test isapprox(ConditionalDistributionForest.compute_forest_weights(forest, X, Xvec, bootstraps), [0.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0; 1.0 0.0; 0.0 1.0; 0.0 0.0; 0.0 0.0])
end


@testset "Computing conditional distribution function" begin
    rng = MersenneTwister(1234)

    n_primes = 5
    Xvec = Array{Float64, 2}(undef, n_primes, 3)
    Xvec[:, 1] .= rand(rng, GeneralizedPareto(0., 1.5, 0.25), n_primes)
    Xvec[:, 2] .= rand(rng, LogNormal(1.1, 0.6), n_primes)
    Xvec[:, 3] .= rand(rng, Gamma(2, 5.0/3), n_primes)

    
    n_samples = 10
    X = Array{Float64, 2}(undef, n_samples, 3)
    X[:, 1] .= rand(rng, GeneralizedPareto(0., 1.5, 0.25), n_samples)
    X[:, 2] .= rand(rng, LogNormal(1.1, 0.6), n_samples)
    X[:, 3] .= rand(rng, Gamma(2, 5.0/3), n_samples)
    
    Y = vec(sum(X, dims=2)) + rand(rng, Normal(0, 2), n_samples)

    
    rng = MersenneTwister(5678)
    forest, bootstraps = build_distribution_forest(Y, X, -1, 1, 1.0, -1, 1, rng = rng, return_bootstrap = true)

    
    Ydiscr = collect(0.5:0.5:10)
    fdr = Array{Float64, 2}(undef, length(Ydiscr), n_primes)
    fdr_ref = [0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 1.0 0.0 1.0; 0.0 0.0 1.0 0.0 1.0; 0.0 0.0 1.0 0.0 1.0; 0.0 0.0 1.0 0.0 1.0; 0.0 0.0 1.0 0.0 1.0; 1.0 0.0 1.0 0.0 1.0; 1.0 0.0 1.0 0.0 1.0; 1.0 0.0 1.0 0.0 1.0; 1.0 0.0 1.0 0.0 1.0]
    
    
    compute_conditional_distribution!(fdr, forest, Y, X, Xvec, Ydiscr, bootstraps)
    @test isapprox(fdr, fdr_ref)

    rng = MersenneTwister(5678)
    compute_conditional_distribution!(fdr, Y, X, Xvec, Ydiscr, -1, 1, 1.0, -1, 1, rng = rng)
    @test isapprox(fdr, fdr_ref)

    @test isapprox(compute_conditional_distribution(forest, Y, X, Xvec, Ydiscr, bootstraps), fdr_ref)

    rng = MersenneTwister(5678)
    @test isapprox(compute_conditional_distribution(Y, X, Xvec, Ydiscr, -1, 1, 1.0, -1, 1, rng = rng), fdr_ref)


    fdr_ref = [0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.5 0.0 0.5; 0.0 0.0 0.5 0.0 0.5; 0.0 0.0 0.5 0.0 0.5; 0.0 0.0 0.5 0.0 0.5; 0.0 0.0 0.5 0.5 0.5; 1.0 0.0 0.5 0.5 0.5; 1.0 0.0 0.5 0.5 0.5; 1.0 0.0 0.5 0.5 0.5; 1.0 0.0 0.5 0.5 0.5]
    
    compute_conditional_distribution!(fdr, forest, Y, X, Xvec, Ydiscr)
    @test isapprox(fdr, fdr_ref)

    @test isapprox(compute_conditional_distribution(forest, Y, X, Xvec, Ydiscr), fdr_ref)
end


@testset "Computing conditional quantiles" begin
    rng = MersenneTwister(1234)

    n_primes = 5
    Xvec = Array{Float64, 2}(undef, n_primes, 3)
    Xvec[:, 1] .= rand(rng, GeneralizedPareto(0., 1.5, 0.25), n_primes)
    Xvec[:, 2] .= rand(rng, LogNormal(1.1, 0.6), n_primes)
    Xvec[:, 3] .= rand(rng, Gamma(2, 5.0/3), n_primes)

    
    n_samples = 10
    X = Array{Float64, 2}(undef, n_samples, 3)
    X[:, 1] .= rand(rng, GeneralizedPareto(0., 1.5, 0.25), n_samples)
    X[:, 2] .= rand(rng, LogNormal(1.1, 0.6), n_samples)
    X[:, 3] .= rand(rng, Gamma(2, 5.0/3), n_samples)
    
    Y = vec(sum(X, dims=2)) + rand(rng, Normal(0, 2), n_samples)

    
    rng = MersenneTwister(5678)
    forest, bootstraps = build_distribution_forest(Y, X, -1, 1, 1.0, -1, 1, rng = rng, return_bootstrap = true)


    alphas = collect(0.1:0.1:0.9)
    quant = Array{Float64, 2}(undef, n_primes, length(alphas))
    quant_ref = [8.450178408561971 8.450178408561971 8.450178408561971 8.450178408561971 8.450178408561971 8.450178408561971 8.450178408561971 8.450178408561971 8.450178408561971; 14.762920820567878 14.762920820567878 14.762920820567878 14.762920820567878 14.762920820567878 14.762920820567878 14.762920820567878 14.762920820567878 14.762920820567878; 5.566834469394174 5.566834469394174 5.566834469394174 5.566834469394174 5.566834469394174 5.566834469394174 5.566834469394174 5.566834469394174 5.566834469394174; 10.922288040267013 10.922288040267013 10.922288040267013 10.922288040267013 10.922288040267013 10.922288040267013 10.922288040267013 10.922288040267013 10.922288040267013; 5.566834469394174 5.566834469394174 5.566834469394174 5.566834469394174 5.566834469394174 5.566834469394174 5.566834469394174 5.566834469394174 5.566834469394174]
    
    
    compute_conditional_quantiles!(quant, forest, Y, X, Xvec, alphas, bootstraps)
    @test isapprox(quant, quant_ref)

    rng = MersenneTwister(5678)
    compute_conditional_quantiles!(quant, Y, X, Xvec, alphas, -1, 1, 1.0, -1, 1, rng = rng)
    @test isapprox(quant, quant_ref)

    @test isapprox(compute_conditional_quantiles(forest, Y, X, Xvec, alphas, bootstraps), quant_ref)

    rng = MersenneTwister(5678)
    @test isapprox(compute_conditional_quantiles(Y, X, Xvec, alphas, -1, 1, 1.0, -1, 1, rng = rng), quant_ref)


    quant_ref = [8.450178408561971 8.450178408561971 8.450178408561971 8.450178408561971 8.450178408561971 8.450178408561971 8.450178408561971 8.450178408561971 8.450178408561971; 14.762920820567878 14.762920820567878 14.762920820567878 14.762920820567878 14.762920820567878 14.762920820567878 14.762920820567878 14.762920820567878 14.762920820567878; 5.566834469394174 5.566834469394174 5.566834469394174 5.566834469394174 10.923414732418404 10.923414732418404 10.923414732418404 10.923414732418404 10.923414732418404; 7.582297789430057 7.582297789430057 7.582297789430057 7.582297789430057 10.922288040267013 10.922288040267013 10.922288040267013 10.922288040267013 10.922288040267013; 5.566834469394174 5.566834469394174 5.566834469394174 5.566834469394174 10.923414732418404 10.923414732418404 10.923414732418404 10.923414732418404 10.923414732418404]
    
    compute_conditional_quantiles!(quant, forest, Y, X, Xvec, alphas)
    @test isapprox(quant, quant_ref)

    @test isapprox(compute_conditional_quantiles(forest, Y, X, Xvec, alphas), quant_ref)
end


@testset "Tools" begin
    rng = MersenneTwister(1234)
    
    n_samples = 10
    X = Array{Float64, 2}(undef, n_samples, 3)
    X[:, 1] .= rand(rng, GeneralizedPareto(0., 1.5, 0.25), n_samples)
    X[:, 2] .= rand(rng, LogNormal(1.1, 0.6), n_samples)
    X[:, 3] .= rand(rng, Gamma(2, 5.0/3), n_samples)
    
    Y = vec(sum(X, dims=2)) + rand(rng, Normal(0, 2), n_samples)

    forest, bootstraps = build_distribution_forest(Y, X, -1, 1, 1.0, -1, 1, rng = rng, return_bootstrap = true)

    tree = forest.trees[1]
    @test ConditionalDistributionForest.list_leafs(tree) == Dict(4 => 1,13 => 4,14 => 5,31 => 7,5 => 2,12 => 3,30 => 6)

    @test ConditionalDistributionForest.get_leaf_of_feature(tree, X) == [4, 6, 6, 7, 4, 7, 1, 5, 2, 3]

    @test ConditionalDistributionForest.get_repartition(tree, X, 1:n_samples) == [[7], [9], [10], [1, 5], [8], [2, 3], [4, 6]]
    @test ConditionalDistributionForest.get_repartition(tree, X, bootstraps[:, 1]) == [[7], [9, 9], [10], [1], [8, 8], [3], [6, 6]]
end
