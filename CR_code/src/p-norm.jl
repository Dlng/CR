include("util.jl")
include("metric.jl")


#TODO optimize
function eval_obj(U, V, X, relThreshold, p)
    # relThreshold = params["relThreshold"]
    # p = params["p"]
    finalRes = 0
    for id in 1:size(X)[1]
        userVec = X[id, :]
        ui = deepcopy(U[:, id])
        userRes = 0
        posItemIdxs = get_pos_items(userVec,relThreshold)
        negItemIdxs = get_neg_items(userVec, relThreshold)
        ni = length(posItemIdxs) + length(negItemIdxs)
        @assert (ni != 0) "PNORM:eval_obj: ni is 0"

        # for negItemIdx in negItemIdxs
        #     vj = deepcopy(V[:, negItemIdx])
        #     curHeight = get_height(vj, ui, V, posItemIdxs)
        #     userRes += curHeight^p
        # end

        # TEMP exp_norm
        for posItemIdx in posItemIdxs
            vk = deepcopy(V[:, posItemIdx])
            curRHeight = get_reverse_height(vk, ui, V, negItemIdxs)
            userRes += curRHeight^p
            # END
        end
        # END TEMP
        finalRes += (1/ni) * userRes

    end
    return finalRes
end


function get_p_norm_gradient_by_user(userVec,ui, V,p, relThreshold)
    # p = params["p"]
    # relThreshold = params["relThreshold"]
    posItemIdxs = get_pos_items(userVec,relThreshold)
    negItemIdxs = get_neg_items(userVec, relThreshold)
    ni = length(posItemIdxs) + length(negItemIdxs)
    @assert (ni != 0) "PNORM:get_p_norm_gradient_by_user: ni is 0"
    res = 0

    # for negItemIdx in negItemIdxs
    #     negItemVec = deepcopy(V[:, negItemIdx])
    #     curHeight = get_height(negItemVec, ui, V, posItemIdxs)
    #     tempSum = 0
    #     for posItemIdx in posItemIdxs
    #         posItemVec = deepcopy(V[:, posItemIdx])
    #         t =  dot(ui, (posItemVec - negItemVec))
    #         tempSum += sigma(t) * (negItemVec - posItemVec)
    #     end
    #     res +=  curHeight ^ (p-1) * tempSum
    # end
    # TEMP
    for posItemIdx in posItemIdxs
        posItemVec = deepcopy(V[:, posItemIdx])
        curRHeight = get_reverse_height(posItemVec, ui, V, negItemIdxs)
        tempSum = 0
        for negItemIdx in negItemIdxs
            negItemVec = deepcopy(V[:, negItemIdx])
            t =  dot(ui, (posItemVec - negItemVec))
            tempSum += sigma(t) * (negItemVec - posItemVec)
        end
        res +=  curRHeight ^ (p-1) * tempSum
    end
    # END TEMP
    return (p / ni)  * res
end

#TODO this phase is SLOW; obs: remapping the item idx significant;y speeds up this
function get_p_norm_gradient_by_item(X, U, V, itemId, p,relThreshold)
    V_dc = deepcopy(V)
    posUsers = get_pos_users(X, itemId,relThreshold)
    negUsers = get_neg_users(X, itemId,relThreshold)
    # debug("get_p_norm_gradient_by_item: size of posUsers $size(posUsers)")
    # debug(size(negUsers))
    finalRes = 0
    for userId in negUsers
        ui = deepcopy(U[:,userId])
        userVec = X[userId, :]
        posItemIdxs = get_pos_items(userVec,relThreshold)
        negItemIdxs = get_neg_items(userVec, relThreshold)
        ni = length(posItemIdxs) + length(negItemIdxs)
        @assert (ni != 0) "PNORM:get_p_norm_gradient_by_item: ni is 0, $itemId, $userId"
        res = 0
        # for negItemIdx in negItemIdxs
        #     negItemVec = deepcopy(V[:, negItemIdx])
        #     curHeight = get_height(negItemVec, ui, V, posItemIdxs)
        #     @assert (curHeight != Inf) "PNORM:get_p_norm_gradient_by_item:curHeight is Inf"
        #
        #     tempSum = 0
        #     for posItemIdx in posItemIdxs
        #         posItemVec = deepcopy(V[:, posItemIdx])
        #         t =  dot(ui, ( posItemVec -negItemVec))
        #         tempSum += sigma(t) * ui
        #     end
        #
        #     res +=  curHeight ^ (p-1) * tempSum
        # end

        # TEMP
        for posItemIdx in posItemIdxs
            posItemVec = deepcopy(V[:, posItemIdx])
            curRHeight = get_reverse_height(posItemVec, ui, V, negItemIdxs)
            @assert (curRHeight != Inf) "PNORM:get_p_norm_gradient_by_item:curRHeight is Inf"
            tempSum = 0
            for negItemIdx in negItemIdxs
                negItemVec = deepcopy(V[:, negItemIdx])
                t =  dot(ui, ( posItemVec -negItemVec))
                tempSum += sigma(t) * ui
            end
            res +=  curRHeight ^ (p-1) * tempSum
        end
        # END TEMP



        finalRes += (p / ni) * res
    end

    # debug("PosUsers get gradient by item: midway gradient is : $finalRes")

    for userId in posUsers
        ui = deepcopy(U[: , userId])
        userVec = X[userId, :]
        posItemIdxs = get_pos_items(userVec,relThreshold)
        negItemIdxs = get_neg_items(userVec,relThreshold)
        ni = length(posItemIdxs) + length(negItemIdxs)
        @assert (ni > 0) "PNORM:get_p_norm_gradient_by_item: ni is 0, $itemId, $userId"

        # debug("pos item Num: $(length(posItemIdxs))")
        # debug("neg item Num: $(length(negItemIdxs))")

        res = 0

        # for negItemIdx in negItemIdxs
        #     negItemVec = deepcopy(V[:, negItemIdx])
        #     curHeight = get_height(negItemVec, ui, V, posItemIdxs)
        #     @assert (curHeight != Inf) "PNORM:get_p_norm_gradient_by_item:curHeight is Inf"
        #     tempSum = 0
        #     for posItemIdx in posItemIdxs
        #         posItemVec = deepcopy(V[:, posItemIdx])
        #         t =  dot(ui, ( posItemVec -negItemVec))
        #         tempSum += sigma(t) * ui
        #         if any(isnan.(tempSum))
        #             debug("PosUsers: curHeight is : $curHeight")
        #             debug("PosUsers: ui is : $ui")
        #             debug("PosUsers: tempSum is : $tempSum")
        #             # debug("PosUsers: res is : $res")
        #             debug(" ")
        #         end
        #     end
        #     res +=  (curHeight ^ (p-1)) * tempSum
        #     if any(isnan.(res))
        #         debug("RES is NAN")
        #         debug("PosUsers: curHeight is : $curHeight")
        #         debug("PosUsers: ui is : $ui")
        #         debug("PosUsers: tempSum is : $tempSum")
        #         debug("PosUsers: res is : $res")
        #         debug(" ")
        #     end
        # end

        #TEMP
        for posItemIdx in posItemIdxs
            posItemVec = deepcopy(V[:, posItemIdx])
            curRHeight = get_reverse_height(posItemVec, ui, V, negItemIdxs)
            @assert (curRHeight != Inf) "PNORM:get_p_norm_gradient_by_item:curRHeight is Inf"
            tempSum = 0
            for negItemIdx in negItemIdxs
                negItemVec = deepcopy(V[:, negItemIdx])
                t =  dot(ui, ( posItemVec -negItemVec))
                tempSum += sigma(t) * ui
                if any(isnan.(tempSum))
                    debug("PosUsers: curRHeight is : $curRHeight")
                    debug("PosUsers: ui is : $ui")
                    debug("PosUsers: tempSum is : $tempSum")
                    # debug("PosUsers: res is : $res")
                    debug(" ")
                end
            end
            res +=  (curRHeight ^ (p-1)) * tempSum
            if any(isnan.(res))
                debug("RES is NAN")
                debug("PosUsers: curRHeight is : $curRHeight")
                debug("PosUsers: ui is : $ui")
                debug("PosUsers: tempSum is : $tempSum")
                debug("PosUsers: res is : $res")
                debug(" ")
            end
        end
        # END

        finalRes -=  (p / ni) * res

    end
    # debug("get_p_norm_gradient_by_item: final gradient is $finalRes")
    return finalRes
end


# return optimized U, V
# ASSUME X is sparse
# ASSUME U, V are non-sparse
# @param Y is the validation set, for early stoping
# @param T is the test set
function p_norm_optimizer(X, U, V, Y, T, learningRate; p = 2, convThreshold=0.0001,
    regval=0.001, relThreshold = 4, iterNum=200, k = 5, metric=2)
    debug("In PNORM")
    isConverge = false
    curEvalVali = 0
    preEvalVali = 0
    userNum = size(U)[2]
    itemNum = size(V)[2]
    # debug("PNORM size of U $(size(U))")
    # debug("PNORM size of V $(size(V))")
    count = 1

    # plotting
    
    plotY_obj = [] # eval obj on training set using updated U V
    plotY_train = [] # eval metric on training set using updated U V
    plotY_eval = [] # eval metric on testing set using updated U V

    for it in 1:iterNum
        debug("Pnorm: On iteration $it")
        debug("Start user phase")
        preEvalVali = curEvalVali
        for i in 1:userNum
            userVec = X[i, :]
            ui = deepcopy(U[:, i])
            # debug("TEST userId i : $i")
            gradient = get_p_norm_gradient_by_user(userVec, ui, V,p, relThreshold)
            ragVal =  regval * ui
            # U[:, i] = (ui- learningRate * (gradient + ragVal))'
            U[:, i] = ui- learningRate * (gradient + ragVal)
            #### TEST
            if all(gradient .== 0)
                debug(U[:, i])
                assert(all(U[:, i] .==0))
                @assert (ui != U[:,i]) "U[:,i] not updated! gradient is 0"
            end
            #### END
            @assert (ui != U[:,i]) "U[:,i] not updated!"
            @assert (any(isnan.(U[:, i])) == false) "ui contains NaN"
        end
        debug("FINISHED user phase")
        # TEST
        debug(any(U .== 0))
        debug(any(U .> 100000))
        debug(any(U .< -1e10))
        # END
        debug("Start item phase")
        for h in 1:itemNum
            vh = deepcopy(V[: , h])
            gradient = get_p_norm_gradient_by_item(X, U, V, h, p, relThreshold)

            ragVal =  regval * vh
            # debug(ragVal)
            # debug(gradient) ## gradient is NaN
            # debug(" ")
            # V[:, h] = (vh - learningRate *(gradient + ragVal))'
            V[:, h] = vh - learningRate *(gradient + ragVal)

            @assert (vh != V[:,h]) "V[:,h] not updated! being $(V[:,h])"
            @assert (any(isnan.(V[:,h])) == false) "vh contains NaN being $(V[:,h])"

        end
        debug("after item phase :V: $(any(V .== 0))")
        debug(any(V .> 100000))
        debug(any(V .< -1e10))

        debug("FINISHED item phase")
        curEvalVali = evaluate(U, V, Y, k = 5, relThreshold = relThreshold, metric=1) # using MAP@5
        curEvalTest = evaluate(U, V, T, k = k, relThreshold = relThreshold, metric=metric)
        curEvalTrain = evaluate(U, V, X, k = k, relThreshold = relThreshold,metric=metric)
        # Test evaluate the loss instead
        curVal_obj = eval_obj(U, V, X, relThreshold, p)

        push!(plotY_eval, curEvalTest)
        push!(plotY_train, curEvalTrain)
        push!(plotY_obj, curVal_obj)
        debug("curEvalTest is $curEvalTest")
        debug("curEvalTrain is $curEvalTrain")
        debug("curVal_obj is $curVal_obj")

        # early stopping
        if it == 1
            debug("RI")
            continue
        else
            diff = (curEvalVali - preEvalVali) # maximizing the map_5
            # diff = ( preVal_obj - curVal_obj) # minimizing the loss
            debug("Diff is $diff")
            if diff <= convThreshold
                isConverge = true
                count = count+1
                break
            end
        end
        count = count+1
        debug("Pnorm: FINISHED iteration $it, curVal_obj is : $curVal_obj")
    end

    debug("Pnorm: EXITED at iteration $count, convergence is :$isConverge")
    debug("PNORN:final plotY_eval :$plotY_eval")
    debug("PNORN:final plotY_obj :$plotY_obj")
    debug("PNORN:final plotY_train :$plotY_train")
    debug("PNORM FINISH")
    curTime = Dates.value(now())
    return U, V, curTime, plotY_eval, plotY_train, plotY_obj
end
