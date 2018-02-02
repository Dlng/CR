using MATLAB

include("util.jl")
include("metric.jl")
include("solve_trace_reg.jl")

#TODO do param selection on regval 1e−4, 1e−3, 1e−2, 1e−1,1


function eval_obj(U, V , X, relThreshold)
    finalRes = 0
    userNum = size(X)[1]
    for id in 1:userNum
        userVec = X[id, :]
        ui = U[:, id]
        userRes = 0
        posItemIdxs = get_pos_items(userVec,relThreshold)
        negItemIdxs = get_neg_items(userVec, relThreshold)
        ni = length(posItemIdxs) + length(negItemIdxs)
        @assert (ni != 0) "RNORM:get_r_norm_gradient_by_user: ni is 0"
        for posItemIdx in posItemIdxs
            vk = deepcopy(V[:, posItemIdx])
            curRHeight = get_reverse_height(vk, ui, V, negItemIdxs)
            userRes += log(1 + curRHeight)
        end
        # #TEMP
        # for negItemIdx in negItemIdxs
        #     vj = deepcopy(V[:, negItemIdx])
        #     curHeight = get_height(vj, ui, V, negItemIdxs)
        #     userRes += log(1 + curHeight)
        # end
        # # END
        finalRes += (1/ni) * userRes

    end
    return finalRes
end



# returns the gradient of loss by each M_im
function get_convex_r_norm_gradient_entry(ui,V, posItemIdxs, negItemIdxs)
    finalGradient = 0
    # userNum = size(X,1)
    ni = length(posItemIdxs) + length(negItemIdxs)
    @assert ni != 0
    for posItemIdx in posItemIdxs
        xk = V[:,posItemIdx]
        curPosItemVal = dot(ui , xk)
        curRHeight = get_reverse_height(xk, ui, V,  negItemIdxs)
        tempSum = 0
        for negItemIdx in negItemIdxs
            curNegItemVal = dot(ui, V[:,negItemIdx])
            delta = curNegItemVal - curPosItemVal
            tempSum += sigma(delta) # move the sign to caller
        end
        finalGradient += 1/(1+curRHeight) * tempSum
    end
    return 1/ni * finalGradient
end


function get_convex_r_norm_gradient( U,V,X,relThreshold)
"""
- since M should not be computed explicitly, for each gradient entry, coord should be recorded
- How to reflect the changes in U, V? or what does solve_trace_reg expect?
"""
    m = size(U,2)
    n = size(V,2)
    Mg = zeros(m,n)
    userNum = size(U,1)
    for userId in 1:userNum
        userVec = X[userId, :]
        # calculate A for user i
        posItemIdxs = get_pos_items(userVec,relThreshold)
        negItemIdxs = get_neg_items(userVec,relThreshold)
        kNum =  length(negItemIdxs)
        jNum = length(posItemIdxs)
        ni = kNum + jNum
        @assert (ni > 0) "PNORM:get_p_norm_gradient_by_item: ni is 0,  $userId"
        ui = deepcopy(U[:,userId])

        # TODO optimize?
        # A = zeros(kNum, jNum) #A remapped indexed
        # for k in 1:kNum
        #     M_ik = M_row_i[posItemIdxs[k]]
        #     for j in 1: jNum
        #         M_ij = M_row_i[negItemIdxs[j]]
        #         A[k,j] = -M_ik + M_ij
        #     end
        # end

        # since gradient of all entries of pos/neg set are the same
        gradient =  get_convex_r_norm_gradient_entry(ui, V, posItemIdxs, negItemIdxs)

        # fill each pos entry in Mg
        for posItemIdx in posItemIdxs
            Mg[userId, posItemIdx] = -gradient
        end
        # fill each neg entry in Mg
        for negItemIdx in negItemIdxs
            Mg[userId, negItemIdx] = gradient
        end
    end
    return Mg

end


function convex_r_norm_loss(U, V , X, relThreshold,)
    """
    function handle of loss to be passed into matlab @lbfgsb
    """
    f = eval_obj(U, V , X, relThreshold)
    G = get_convex_r_norm_gradient( U,V,X,relThreshold)
    m = size(U,2)
    n = size(V,2)
    dims = [m,n]
    lambda = 10
    opts = Dict()
    opts["relThreshold"] = relThreshold
    opts["init_rank"] = 2
    opts["max_iter"] = 10 # the number of rank, 10 is the val in CR with push
    opts["use_local"] = true # ??
    opts["verbose"] = true
end


"""
@PARAM: M: the replacement of U * V'; init with zeros
"""
# function convex_r_norm_optimizer(X, M, Y, T, learningRate; convThreshold=0.0001,
#     regval=0.001, relThreshold = 4, rank=10, k = 5, metric=2)
#     userNum = size(X,1)
#     isConverge = false
#     curEvalVali = 0
#     preEvalVali = 0
#     # debug("PNORM size of U $(size(U))")
#     # debug("PNORM size of V $(size(V))")
#     count = 1
#
#     # plotting
#     plotX = []
#     plotY_obj = [] # eval obj on training set using updated U V
#     plotY_train = [] # eval metric on training set using updated U V
#     plotY_eval = [] # eval metric on testing set using updated U V
#
#     for it in 1:rank
#         debug("Pnorm: On iteration $it")
#         debug("Start user phase")
#         preEvalVali = curEvalVali
#
#         # get gradient matrix Mg
#         Mg = zeros(size(M))
#
#         for userId in 1:userNum
#             userVec = X[userId, :]
#             # calculate A for user i
#             posItemIdxs = get_pos_items(userVec,relThreshold)
#             negItemIdxs = get_neg_items(userVec,relThreshold)
#             kNum =  length(negItemIdxs)
#             jNum = length(posItemIdxs)
#             ni = kNum + jNum
#             @assert (ni > 0) "PNORM:get_p_norm_gradient_by_item: ni is 0,  $userId"
#
#             # TODO optimize?
#             M_row_i = deepcopy(M[userId,:]) # M index is the same as X
#             # A = zeros(kNum, jNum) #A remapped indexed
#             # for k in 1:kNum
#             #     M_ik = M_row_i[posItemIdxs[k]]
#             #     for j in 1: jNum
#             #         M_ij = M_row_i[negItemIdxs[j]]
#             #         A[k,j] = -M_ik + M_ij
#             #     end
#             # end
#
#             # since gradient of all entries of pos/neg set are the same
#             gradient =  get_convex_r_norm_gradient(M_row_i, posItemIdxs, negItemIdxs, relThreshold)
#
#             # fill each pos entry in Mg
#             for posItemIdx in posItemIdxs
#                 Mg[userId, posItemIdx] = -gradient
#             end
#             # fill each neg entry in Mg
#             for negItemIdx in negItemIdxs
#                 Mg[userId, negItemIdx] = gradient
#             end
#
#         end
#         # feed Mg and loss functin in to GCG, returns Mc
#
#         # update M with Mc
#     end
#
# end
