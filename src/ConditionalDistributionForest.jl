module ConditionalDistributionForest

using Random
using DecisionTree

export compute_conditional_quantiles!, compute_conditional_quantiles, compute_conditional_distribution, build_forest


include("tree_tools.jl")


function build_forest(
    labels              :: Vector{T},
    features            :: Matrix{S},
    n_subfeatures       = -1,
    n_trees             = 10,
    partial_sampling    = 0.7,
    max_depth           = -1,
    min_samples_leaf    = 1,
    min_samples_split   = 2,
    min_purity_increase = 0.0;
    rng                 = Random.GLOBAL_RNG,
    return_bootstrap    = false) where {S, T}

    if return_bootstrap
        return build_forest_bootstraps(labels,
                                       features,
                                       n_subfeatures,
                                       n_trees,
                                       partial_sampling,
                                       max_depth,
                                       min_samples_leaf,
                                       min_samples_split,
                                       min_purity_increase,
                                       rng = rng)
    else
        return build_forest(labels,
                            features,
                            n_subfeatures,
                            n_trees,
                            partial_sampling,
                            max_depth,
                            min_samples_leaf,
                            min_samples_split,
                            min_purity_increase,
                            rng = rng)
    end
end


function compute_conditional_quantiles!(quant::AbstractMatrix{T},
                                        forest::DecisionTree.Ensemble{S, T},
                                        Y::Vector{T},
                                        X::Matrix{S},
                                        Xvec::Matrix{S},
                                        quantiles::Vector{U},
                                        bootstraps = nothing) where {S, T, U <:AbstractFloat}
    n_prime = size(Xvec, 1)
    n_quantiles = length(quantiles)
    
    weights = compute_forest_weights(forest, X, Xvec, bootstraps)

    perm_Y = sortperm(Y)
    for iquantile in 1:n_quantiles
        for i in 1:n_prime
            compt = 0
            sum_weights = 0.
            while(sum_weights <= quantiles[iquantile])
                compt += 1
                sum_weights += weights[perm_Y[compt], i]
            end
            
            quant[i, iquantile] = Y[perm_Y[compt]]
        end
    end

    return nothing
end
    

function compute_conditional_quantiles!(quant::AbstractMatrix{T},
                                        Y::Vector{T},
                                        X::Matrix{S},
                                        Xvec::Matrix{S},
                                        quantiles::Vector{U},
                                        n_subfeatures       = -1,
                                        n_trees             = 10,
                                        partial_sampling    = 0.7,
                                        max_depth           = -1,
                                        min_samples_leaf    = 1,
                                        min_samples_split   = 2,
                                        min_purity_increase = 0.0,
                                        rng                 = Random.GLOBAL_RNG) where {S, T, U <:AbstractFloat}
    forest, bootstraps = build_forest_bootstraps(Y, X, n_subfeatures, n_trees, partial_sampling, max_depth, min_samples_leaf, min_samples_split, min_purity_increase, rng = rng)

    compute_conditional_quantiles!(quant, forest, Y, X, Xvec, quantiles, bootstraps)

    return nothing
end



function compute_conditional_quantiles(forest::DecisionTree.Ensemble{S, T},
                                       Y::Vector{T},
                                       X::Matrix{S},
                                       Xvec::Matrix{S},
                                       quantiles::Vector{U},
                                       bootstraps = nothing) where {S, T, U <:AbstractFloat}
    quant = Matrix{T}(undef, size(X, 1), length(quantiles))
    compute_conditional_quantiles!(quant, forest, Y, X, Xvec, quantiles, bootstraps)
    return quant
end


function compute_conditional_quantiles!(Y::Vector{T},
                                        X::Matrix{S},
                                        Xvec::Matrix{S},
                                        quantiles::Vector{U},
                                        n_subfeatures       = -1,
                                        n_trees             = 10,
                                        partial_sampling    = 0.7,
                                        max_depth           = -1,
                                        min_samples_leaf    = 1,
                                        min_samples_split   = 2,
                                        min_purity_increase = 0.0,
                                        rng                 = Random.GLOBAL_RNG) where {S, T, U <:AbstractFloat}
    forest, bootstraps = build_forest_bootstraps(Y, X, n_subfeatures, n_trees, partial_sampling, max_depth, min_samples_leaf, min_samples_split, min_purity_increase, rng = rng)

    return compute_conditional_quantiles(forest, Y, X, Xvec, quantiles, bootstraps)
end



function build_forest_bootstraps(
    labels              :: Vector{T},
    features            :: Matrix{S},
    n_subfeatures       = -1,
    n_trees             = 10,
    partial_sampling    = 0.7,
    max_depth           = -1,
    min_samples_leaf    = 1,
    min_samples_split   = 2,
    min_purity_increase = 0.0;
    rng                 = Random.GLOBAL_RNG) where {S, T}
    
    if n_trees < 1
        throw("the number of trees must be >= 1")
    end
    if !(0.0 < partial_sampling <= 1.0)
        throw("partial_sampling must be in the range (0,1]")
    end

    if n_subfeatures == -1
        n_features = size(features, 2)
        n_subfeatures = round(Int, sqrt(n_features))
    end

    t_samples = length(labels)
    n_samples = floor(Int, partial_sampling * t_samples)

    rngs = DecisionTree.mk_rng(rng)::Random.AbstractRNG
    forest = Vector{DecisionTree.LeafOrNode{S, T}}(undef, n_trees)

    bootstraps = Array{Int64, 2}(undef, n_samples, n_trees)
    Threads.@threads for i in 1:n_trees
        bootstraps[:, i]  .= rand(rngs, 1:t_samples, n_samples)
        forest[i] = build_tree(
            labels[bootstraps[:, i]],
            features[bootstraps[:, i], :],
            n_subfeatures,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            min_purity_increase,
            rng = rngs)
    end

    return Ensemble{S, T}(forest), bootstraps
end




function compute_forest_weights(model, X, Xvec, bootstraps)
    n_samples = size(X, 1)
    n_prime = size(Xvec, 1)
    n_trees = length(model.trees)
    
    weights = zeros(n_samples, n_prime)
    for l in 1:n_trees
	    leafids = get_leaf_of_feature(model.trees[l], Xvec)
        repartition = get_repartition(model.trees[l], X, bootstraps[:, l])
        
	    for i in 1:n_prime
            val = 1 / length(repartition[leafids[i]])
	        for j in repartition[leafids[i]]
	            weights[j, i] += val
	        end
	    end
    end
    
    weights ./= n_trees
    return weights
end 


function compute_forest_weights(model, X, Xvec, ::Nothing)
    n_samples = size(X, 1)
    n_prime = size(Xvec, 1)
    n_trees = length(model.trees)
    
    weights = zeros(n_samples, n_prime)
    for l in 1:n_trees
	    leafids = get_leaf_of_feature(model.trees[l], Xvec)
        repartition = get_repartition(model.trees[l], X, 1:n_samples)
        
	    for i in 1:n_prime
            val = 1 / length(repartition[leafids[i]])
	        for j in repartition[leafids[i]]
	            weights[j, i] += val
	        end
	    end
    end
    
    weights ./= n_trees
    return weights
end 

end # module
