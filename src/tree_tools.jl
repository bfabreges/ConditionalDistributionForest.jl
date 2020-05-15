using DecisionTree


## Function that list all the leafs in a tree and the number of partition in each leaf
## It returns a dictionary.


function list_leafs!(leafs::Dict{Int64, Int64}, leaf::Leaf{T}, leafid::Int64, index::Int64) where {T}
    leafs[leafid] = index
    return index + 1
end

function list_leafs!(leafs::Dict{Int64, Int64}, node::Node{S, T}, leafid::Int64, index::Int64) where {S, T}
    index = list_leafs!(leafs, node.left, leafid << 1, index)
    index = list_leafs!(leafs, node.right, (leafid << 1) + 1, index)
    return index
end

function list_leafs(tree::DecisionTree.LeafOrNode{S, T}) where {S, T}
    leafs = Dict{Int64, Int64}()
    list_leafs!(leafs, tree, 1, 1)
    return leafs
end


## Function that returns a dictionnary
## Keys are the leafs names
## Values are the vectors of id of the features in the leaf


get_leaf_of_feature(leaf::Leaf{T}, feature::AbstractVector{S}, leafid::Int64) where {S, T} = leafid

function get_leaf_of_feature(tree::Node{S, T}, feature::AbstractVector{S}, leafid::Int64) where {S, T}
    if feature[tree.featid] < tree.featval
        return get_leaf_of_feature(tree.left, feature, leafid << 1)
    else
        return get_leaf_of_feature(tree.right, feature, (leafid << 1) + 1)
    end
end

function get_leaf_of_feature(tree::DecisionTree.LeafOrNode{S, T}, feature::AbstractVector{S}) where {S, T}
    leafnames = list_leafs(tree)
    return leafnames[get_leaf_of_feature(tree, feature, 1)]
end

function get_leaf_of_feature(tree::DecisionTree.LeafOrNode{S, T}, features::AbstractMatrix{S}) where {S, T}
    leafnames = list_leafs(tree)    

    N = size(features, 1)
    leafs = Array{Int64, 1}(undef, N)
    for i in 1:N
        leafs[i] = leafnames[get_leaf_of_feature(tree, view(features, i, :), 1)]
    end

    return leafs
end

function get_repartition(tree::DecisionTree.LeafOrNode{S, T}, features::AbstractMatrix{S}, bootstrap::AbstractVector{Int64}) where {S, T}
    nleafs = length(tree)

    leafs_of_features = get_leaf_of_feature(tree, features[bootstrap, :])
    
    count_partition = zeros(Int64, nleafs)
    for idleaf in leafs_of_features
        count_partition[idleaf] += 1
    end

    partition = Array{Array{Int64, 1}, 1}(undef, nleafs)
    for ileaf in 1:nleafs
        partition[ileaf] = Array{Int64, 1}(undef, count_partition[ileaf])
        count_partition[ileaf] = 0
    end

    for i in 1:length(leafs_of_features)
        ileaf = leafs_of_features[i]
        count_partition[ileaf] += 1
        partition[ileaf][count_partition[ileaf]] = bootstrap[i]
    end 
    
    return partition
end
