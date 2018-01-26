include("util.jl")
include("metric.jl")

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


function get_r_norm_gradient_by_user(userVec,ui, V, relThreshold)

    posItemIdxs = get_pos_items(userVec,relThreshold)
    negItemIdxs = get_neg_items(userVec, relThreshold)
    ni = length(posItemIdxs) + length(negItemIdxs)
    @assert (ni != 0) "RNORM:get_r_norm_gradient_by_user: ni is 0"
    # debug("size of posItems $(length(posItems))")
    res = 0
    for posItemIdx in posItemIdxs
        posItemVec = deepcopy(V[:, posItemIdx])
        curRHeight = get_reverse_height(posItemVec, ui, V, negItemIdxs)
        tempSum = 0
        for negItemIdx in negItemIdxs
            negItemVec = deepcopy(V[:, negItemIdx])
            t =  dot(ui, (posItemVec - negItemVec))
            tempSum += sigma(t) * (negItemVec - posItemVec)
        end
        res +=  1/(1 + curRHeight) * tempSum
    end

    # #TEMP
    # for negItemIdx in negItemIdxs
    #     negItemVec = deepcopy(V[:, negItemIdx])
    #     curHeight = get_height(negItemVec, ui, V, posItemIdxs)
    #     tempSum = 0
    #     for posItemIdx in posItemIdxs
    #         posItemVec = deepcopy(V[:, posItemIdx])
    #         t =  dot(ui, (posItemVec - negItemVec))
    #         tempSum += sigma(t) * (negItemVec - posItemVec)
    #     end
    #     res +=  1/(1 + curHeight) * tempSum
    # end
    # #END


    return (1 / ni)  * res
end


function get_r_norm_gradient_by_item(X, U, V, itemId, relThreshold)
    posUsers = get_pos_users(X, itemId,relThreshold)
    negUsers = get_neg_users(X, itemId,relThreshold)
    # debug(size(posUsers))
    # debug(size(negUsers))
    finalRes = 0
    for userId in negUsers
        ui = U[:,userId]
        userVec = X[userId, :]
        posItemIdxs = get_pos_items(userVec,relThreshold)
        negItemIdxs = get_neg_items(userVec, relThreshold)
        ni = length(posItemIdxs) + length(negItemIdxs)
        @assert (ni != 0) "RNORM:get_p_norm_gradient_by_item: ni is 0, $itemId, $userId"
        res = 0
        for posItemIdx in posItemIdxs
            posItemVec = deepcopy(V[:, posItemIdx])
            curRHeight = get_reverse_height(posItemVec, ui, V, negItemIdxs)
            @assert (curRHeight != Inf) "RNORM:get_r_norm_gradient_by_item:curRHeight is Inf"
            tempSum = 0
            for negItemIdx in negItemIdxs
                negItemVec = deepcopy(V[:, negItemIdx])
                t =  dot(ui, ( posItemVec -negItemVec))
                tempSum += sigma(t) * ui
            end

            res +=  1/(1 + curRHeight)  * tempSum
        end
        # #TEMP
        # for negItemIdx in negItemIdxs
        #     negItemVec = deepcopy(V[:, negItemIdx])
        #     curHeight = get_height(negItemVec, ui, V, posItemIdxs)
        #     @assert (curHeight != Inf) "RNORM:get_r_norm_gradient_by_item:curRHeight is Inf"
        #     tempSum = 0
        #     for posItemIdx in posItemIdxs
        #         posItemVec = deepcopy(V[:, posItemIdx])
        #         t =  dot(ui, ( posItemVec -negItemVec))
        #         tempSum += sigma(t) * ui
        #     end
        #     res +=  1/(1 + curHeight)  * tempSum
        # end
        # # END



        finalRes += (1 / ni) * res
    end

    # debug("PosUsers: midway gradient is : $finalRes")

    for userId in posUsers
        ui = U[: , userId]
        userVec = X[userId, :]
        posItemIdxs = get_pos_items(userVec,relThreshold)
        negItemIdxs = get_neg_items(userVec,relThreshold)
        ni = length(posItemIdxs) + length(negItemIdxs)
        @assert (ni != 0) "PNORM:get_p_norm_gradient_by_item: ni is 0, $itemId, $userId"
        res = 0
        for posItemIdx in posItemIdxs
            posItemVec = deepcopy(V[:, posItemIdx])
            curRHeight = get_reverse_height(posItemVec, ui, V, negItemIdxs)
            @assert (curRHeight != Inf) "RNORM:get_r_norm_gradient_by_item:curRHeight is Inf"
            tempSum = 0
            for negItemIdx in negItemIdxs
                negItemVec = deepcopy(V[:, negItemIdx])
                t =  dot(ui, ( posItemVec -negItemVec))
                tempSum += sigma(t) * ui
                # TEST
                if any(isnan.(tempSum))
                    debug("PosUsers: curRHeight is : $curRHeight")
                    debug("PosUsers: ui is : $ui")
                    debug("PosUsers: tempSum is : $tempSum")
                    # debug("PosUsers: res is : $res")
                    debug(" ")
                end

                # END
            end
            res +=  1/(1+curRHeight)  * tempSum
            # TEST
            if any(isnan.(res))
                debug("RES is NAN")
                debug("PosUsers: curRHeight is : $curRHeight")
                debug("PosUsers: ui is : $ui")
                debug("PosUsers: tempSum is : $tempSum")
                debug("PosUsers: res is : $res")
                debug(" ")
            end
            # END
        end
        # # TEMP
        # for negItemIdx in negItemIdxs
        #     negItemVec = deepcopy(V[:, negItemIdx])
        #     curHeight = get_height(negItemVec, ui, V, posItemIdxs)
        #     @assert (curHeight != Inf) "RNORM:get_r_norm_gradient_by_item:curHeight is Inf"
        #     tempSum = 0
        #     for posItemIdx in posItemIdxs
        #         posItemVec = deepcopy(V[:, posItemIdx])
        #         t =  dot(ui, ( posItemVec -negItemVec))
        #         tempSum += sigma(t) * ui
        #         # TEST
        #         if any(isnan.(tempSum))
        #             debug("PosUsers: curHeight is : $curHeight")
        #             debug("PosUsers: ui is : $ui")
        #             debug("PosUsers: tempSum is : $tempSum")
        #             # debug("PosUsers: res is : $res")
        #             debug(" ")
        #         end
        #
        #         # END
        #     end
        #     res +=  1/(1+curHeight)  * tempSum
        #     # TEST
        #     if any(isnan.(res))
        #         debug("RES is NAN")
        #         debug("PosUsers: curHeight is : $curHeight")
        #         debug("PosUsers: ui is : $ui")
        #         debug("PosUsers: tempSum is : $tempSum")
        #         debug("PosUsers: res is : $res")
        #         debug(" ")
        #     end
        #     # END
        # end
        # # END


        finalRes -=  (1 / ni) * res
        # TEST
        if any(isnan.(finalRes))
            debug("PosUsers: ui is : $ui")
            debug("PosUsers: res is : $res")
            debug("PosUsers: ni is : $ni")
            debug("PosUsers: finalres is : $finalRes")
            debug(" ")
        end
        # END
    end
    # debug("final gradient is $finalRes")
    return finalRes
end


# return optimized U, V
# ASSUME X is sparse
# ASSUME U, V are non-sparse
# @param Y is the validation set
function r_norm_optimizer(X, U, V, Y, T; learningRate=0.0001, convThreshold=0.0001,regval=regval,
    relThreshold = 4, iterNum=200, k=5, metric = 2)
    debug("In RNORM")
    isConverge = false
    curEvalVali = 0
    preEvalVali = 0
    userNum = size(U)[2]
    itemNum = size(V)[2]
    count = 1

    # plotting
    plotX = []
    plotY_obj = []
    plotY_train = []
    plotY_eval = []
    for it in 1:iterNum
        debug("RNORM: On iteration $it")
        debug("RNORM: Start user phase")
        preEvalVali = curEvalVali
        for i in 1:userNum
            userVec = X[i, :]
            ui = U[:, i]
            #### TEST
            gradient = get_r_norm_gradient_by_user(userVec, ui, V, relThreshold)
            ragVal =  regval * ui
            U[:, i] = ui- learningRate * (gradient + ragVal)
            if all(gradient .== 0)
                debug(U[:, i])
                assert(all(U[:, i] .==0))
                @assert (ui != U[:,i]) "U[:,i] not updated! gradient is 0"
            end
            #### END
            # U[:, i] = (ui - (learningRate * get_p_norm_gradient_by_user(userVec, ui, U, V,p, relThreshold) + regval*ui))'
            @assert (ui != U[:,i]) "U[:,i] not updated!"
            @assert (any(isnan.(U[:, i])) == false) "ui contains NaN"
            # debug(learningRate)
            # debug(regval)
            # debug(temp1 ./ U[:,i])
            # debug(temp1)
            # debug(U[:,i])
        end
        debug("RNORM: FINISHED user phase")
        # TEST
        debug(any(U .== 0))
        debug(any(U .> 100000))
        debug(any(U .< -1e10))
        # END
        debug("RNORM: Start item phase")
        for h in 1:itemNum
            itemId = h
            vh = V[: , h]
            # #### TEST
            gradient = get_r_norm_gradient_by_item(X, U, V, itemId,  relThreshold)
            ragVal =  regval * vh
            # # debug(vh./gradient)
            # # debug(ragVal)
            ########## CHECK Matrix op gives NaN
            # debug(V[:,h])
            # debug(gradient) ## gradient is NaN
            V[:, h] = vh - learningRate *(gradient + ragVal)
            # debug("learning rate is $learningRate")
            # debug(ragVal)
            # END
            # V[: , h] = vh - learningRate * (get_p_norm_gradient_by_item(X, U, V, itemId, p, relThreshold) + regval*vh)
            @assert (vh != V[:,h]) "V[:,h] not updated! being $(V[:,h])"
            @assert (any(isnan.(V[:,h])) == false) "vh contains NaN being $(V[:,h])"
        end
        debug("RNORM: FINISHED item phase")

        curEvalVali = evaluate(U, V, Y, k = 5, relThreshold = relThreshold, metric=1) # using MAP@5
        curEvalTest = evaluate(U, V, T, k = k, relThreshold = relThreshold, metric=metric)
        curEvalTrain = evaluate(U, V, X, k = k, relThreshold = relThreshold,metric=metric)
        # Test evaluate the loss instead
        curVal_obj = eval_obj(U, V, X, relThreshold)

        push!(plotY_eval, curEvalTest)
        push!(plotY_train, curEvalTrain)
        push!(plotY_obj, curVal_obj)
        debug("curEvalTest is $curEvalTest")
        debug("curEvalTrain is $curEvalTrain")
        debug("curVal_obj is $curVal_obj")
        if it == 1
            debug("RI")
            continue
        # TEST run full iteration
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
        debug("RNORM: FINISHED iteration $it, curVal_obj is : $curVal_obj")
    end

    debug("RNORM: EXITED at iteration $count, convergence is :$isConverge")


    debug("PARAMS")
    debug("learningRate: $learningRate, convThreshold: $convThreshold,
    regval:$regval, relThreshold:$relThreshold, iterNum:$iterNum, k:$k")
    debug("END PARAMS")


    debug("RNORN:final plotY_eval :$plotY_eval")
    debug("RNORN:final plotY_obj :$plotY_obj")
    debug("RNORN:final plotY_train :$plotY_train")
    return U, V,  plotY_eval, plotY_train, plotY_obj
end
