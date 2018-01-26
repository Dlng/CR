include("util.jl")
include("metric.jl")


function eval_obj(M, X, relThreshold, p)
    finalRes = 0
    userNum = size(X)[1]
    for userId in 1:userNum
        userVec = X[userId, :]
        userRes = 0
        posItemIdxs = get_pos_items(userVec,relThreshold)
        negItemIdxs = get_neg_items(userVec, relThreshold)
        ni = length(posItemIdxs) + length(negItemIdxs)
        @assert (ni != 0) "RNORM:get_r_norm_gradient_by_user: ni is 0"
        for posItemIdx in posItemIdxs
            curRHeight = get_reverse_height_convex(M, userId, posItemIdx, negItemIdxs)
            userRes += log(1+ curRHeight)
        end
        finalRes += (1/ni) * userRes
    end
    return finalRes
end
