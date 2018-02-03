include("util.jl")
include("metric.jl")
using Logging

#TODO do param selection on regval 1e−4, 1e−3, 1e−2, 1e−1,1

#TODO optimize
function eval_obj(U, V, X, relThreshold, p)
    finalRes = 0
    for id in 1:size(X)[1]
        userVec = X[id, :]
        ui = deepcopy(U[:, id])
        userRes = 0
        posItemIdxs = get_pos_items(userVec,relThreshold)
        negItemIdxs = get_neg_items(userVec, relThreshold)
        ni = length(posItemIdxs) + length(negItemIdxs)
        @assert (ni != 0) "PNORM:eval_obj: ni is 0"

        for negItemIdx in negItemIdxs
            vj = deepcopy(V[:, negItemIdx])
            curHeight = get_height(vj, ui, V, posItemIdxs)
            userRes += curHeight^p
        end

        # # TEMP exp_norm
        # for posItemIdx in posItemIdxs
        #     vk = deepcopy(V[:, posItemIdx])
        #     curRHeight = get_reverse_height(vk, ui, V, negItemIdxs)
        #     userRes += curRHeight^p
        #     # END
        # end
        # # END TEMP
        finalRes += (1/ni) * userRes

    end
    return finalRes
end


# returns the gradient of loss by each M_im
# the gradient for all pos / neg items are the same
# use ui, V to replace userRowM
function get_convex_p_norm_gradient(ui,V, posItemIdxs, negItemIdxs,p)
    @assert "p" in keys(params)
    p = params["p"]
    finalGradient = 0
    ni = length(posItemIdxs) + length(negItemIdxs)
    @assert ni != 0
    for negItemIdx in negItemIdxs
        xj = V[:,negItemIdx]
        curNegItemVal = dot(ui, xj)
        curHeight = get_height(xj , ui, V,  posItemIdxs)
        tempSum = 0
        for posItemIdx in posItemIdxs
            curPosItemVal = dot(ui , V[:,posItemIdx])
            delta = curNegItemVal - curPosItemVal
            tempSum += sigma(delta) # move the sign to caller
        end
        finalGradient += curHeight^(p-1) * tempSum
    end
    return p/ni * finalGradient
end


"""
@PARAM: U, V; for filling M, which is too large to be computed explicitly
"""
function convex_p_norm_optimizer(X, U, V, Y, T, learningRate; p = 2, convThreshold=0.0001,
    regval=0.001, relThreshold = 4, rank=10, k = 5, metric=2)
    
    userNum = size(X,1)
    isConverge = false
    curEvalVali = 0
    preEvalVali = 0
    # debug("PNORM size of U $(size(U))")
    # debug("PNORM size of V $(size(V))")
    count = 1

    # plotting
    plotX = []
    plotY_obj = [] # eval obj on training set using updated U V
    plotY_train = [] # eval metric on training set using updated U V
    plotY_eval = [] # eval metric on testing set using updated U V

    for it in 1:rank
        debug("Pnorm: On iteration $it")
        debug("Start user phase")
        preEvalVali = curEvalVali

        # get gradient matrix Mg
        Mg = zeros(size(M))

        for userId in 1:userNum
            userVec = X[userId, :]
            # calculate A for user i
            posItemIdxs = get_pos_items(userVec,relThreshold)
            negItemIdxs = get_neg_items(userVec,relThreshold)
            kNum =  length(negItemIdxs)
            jNum = length(posItemIdxs)
            ni = kNum + jNum
            @assert (ni > 0) "PNORM:get_p_norm_gradient_by_item: ni is 0,  $userId"

            # TODO optimize?
            M_row_i = deepcopy(M[userId,:]) # M index is the same as X
            # A = zeros(kNum, jNum) #A remapped indexed
            # for k in 1:kNum
            #     M_ik = M_row_i[posItemIdxs[k]]
            #     for j in 1: jNum
            #         M_ij = M_row_i[negItemIdxs[j]]
            #         A[k,j] = -M_ik + M_ij
            #     end
            # end

            # since gradient of all entries of pos/neg set are the same
            gradient =  get_convex_p_norm_gradient(M_row_i, posItemIdxs, negItemIdxs, p)

            # fill each pos entry in Mg
            for posItemIdx in posItemIdxs
                Mg[userId, posItemIdx] = -gradient
            end
            # fill each neg entry in Mg
            for negItemIdx in negItemIdxs
                Mg[userId, negItemIdx] = gradient
            end

        end
        # feed Mg and loss functin in to GCG, returns Mc

        # update M with Mc
    end

end
