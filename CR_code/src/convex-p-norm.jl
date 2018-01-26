include("util.jl")
include("metric.jl")


#TODO do param selection on regval 1e−4, 1e−3, 1e−2, 1e−1,1

# TODO TEST
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
        for negItemIdx in negItemIdxs
            curHeight = get_height_convex(M, userId, negItemIdx, posItemIdxs)
            userRes += curHeight^p
        end
        finalRes += (1/ni) * userRes

    end
    return finalRes
end

"""
@ param: isPos : if current item is positive
@ param: curItemVal: the current entry in M (M_im)
@ param: sumItemIdxs: the set of index to sum over (i.e. negItemIdxs if curItem is pos; vice versa)
"""




# returns the gradient of loss by each M_im
# TODO
function get_convex_p_norm_gradient(M, userId, posItemIdxs, negItemIdxs)
    finalGradient = 0
    userNum = size(X)[1]  
    for userId in 1:userNum
        userGradient = 0
        userVec = X[userId, :]
        posItemIdxs = get_pos_items(userVec,relThreshold)
        negItemIdxs = get_neg_items(userVec, relThreshold)
        userVecM = M[userId, :]
        res = 0
        for negItemIdx in negItemIdxs
            # get neg user
                # get
            # get pos user

        end

        for posItemIdx in posItemIdxs

        end

    end
    return finalGradient
end

# """
# @PARAM: M: the replacement of U * V'; init with zeros
# """
# function convex_p_norm_optimizer(X, M, Y, T, learningRate; p = 2, convThreshold=0.0001,
#     regval=0.001, relThreshold = 4, iterNum=200, k = 5, metric=2)
#
#     isConverge = false
#     curEvalVali = 0
#     preEvalVali = 0
#     userNum = size(U)[2]
#     itemNum = size(V)[2]
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
#         for i in 1:userNum
#             userVec = X[i, :]
#             # calculate A for user i
#             posItemIdxs = get_pos_items(userVec,relThreshold)
#             negItemIdxs = get_neg_items(userVec,relThreshold)
#             kNum =  length(negItemIdxs)
#             jNum = length(posItemIdxs)
#             ni = kNum + jNum
#             @assert (ni > 0) "PNORM:get_p_norm_gradient_by_item: ni is 0, $itemId, $userId"
#
#             # TODO optimize?
#             M_row_i = M[i,:] # M index is the same as X
#             A = zeros(kNum, jNum) #A remapped indexed
#             for k in 1:kNum
#                 M_ik = M_row_i[posItemIdxs[k]]
#                 for j in 1: jNum
#                     M_ij = M_row_i[negItemIdxs[j]]
#                     A[k,j] = -M_ik + M_ij
#                 end
#             end
#
#
#
#         end
#
# end
