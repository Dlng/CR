include("util.jl")
include("metric.jl")

function eval_obj(U, V , X, relThreshold)
    finalRes = 0
    userNum = size(X)[1]
    for id in 1:userNum
        userVec = X[id, :]
        ui = U[:, id]
        ni = length(userVec)
        @assert (ni != 0) "RNORM:get_r_norm_gradient_by_user: ni is 0"
        userRes = 0
        posItemIdxs = get_pos_items(userVec,relThreshold)
        negItemIdxs = get_neg_items(userVec, relThreshold)

        for posItemIdx in posItemIdxs
            vk = V[:, posItemIdx]
            curRHeight = get_reverse_height(vk, ui, V, negItemIdxs)
            userRes += log(1 + curRHeight)
        end
        finalRes += (1/ni) * userRes

    end
    return finalRes
end


function get_r_norm_gradient_by_user(userVec,ui, V, relThreshold)
    ni = length(userVec)
    @assert (ni != 0) "RNORM:get_r_norm_gradient_by_user: ni is 0"
    posItemIdxs = get_pos_items(userVec,relThreshold)
    negItemIdxs = get_neg_items(userVec, relThreshold)
    # println("size of posItems $(length(posItems))")
    res = 0
    for posItemIdx in posItemIdxs
        posItemVec = V[:, posItemIdx]
        curRHeight = get_reverse_height(posItemVec, ui, V, negItemIdxs)
        tempSum = 0
        for negItemIdx in negItemIdxs
            negItemVec = V[:, negItemIdx]
            t =  dot(ui, (posItemVec - negItemVec))
            tempSum += sigma(t) * (negItemVec - posItemVec)
        end
        res +=  1/(1 + curRHeight) * tempSum
        #TEST
        if all(res .== 0)

            println(size(posItemIdxs))
            println(size(negItemIdxs))

            println(curRHeight)
            println(tempSum)
            println(" ")
        end
        #END
    end
    return (1 / ni)  * res
end


function get_r_norm_gradient_by_item(X, U, V, itemId, relThreshold)
    posUsers = get_pos_users(X, itemId,relThreshold)
    negUsers = get_neg_users(X, itemId,relThreshold)
    # println(size(posUsers))
    # println(size(negUsers))
    finalRes = 0
    for userId in negUsers
        ui = U[:,userId]
        userVec = X[userId, :]
        ni = size(userVec)[1]
        @assert (ni != 0) "RNORM:get_p_norm_gradient_by_item: ni is 0, $itemId, $userId"
        posItemIdxs = get_pos_items(userVec,relThreshold)
        negItemIdxs = get_neg_items(userVec, relThreshold)
        res = 0
        for negItemIdx in negItemIdxs
            negItemVec = V[:, negItemIdx]
            curRHeight = get_reverse_height(negItemVec, ui, V, posItemIdxs)
            @assert (curRHeight != Inf) "RNORM:get_r_norm_gradient_by_item:curRHeight is Inf"
            tempSum = 0
            for posItemIdx in posItemIdxs
                posItemVec = V[:, posItemIdx]
                t =  dot(ui, ( posItemVec -negItemVec))
                tempSum += sigma(t) * ui
                # println("g-item: dot product is : $t")
                # println("g-item: posItemVec is : $posItemVec")
            end

            res +=  1/(1 + curRHeight)  * tempSum
            # println(" ")
            # println("g-item: ui is : $ui")
            # println("g-item: negItemVec is : $negItemVec")
            # println("g-item: tempSum is : $tempSum")
            # println("g-item: curHeight is : $curHeight")
            # println("g-item: res is : $res")
            # println(" ")
            # println(" ")
        end
        finalRes += (1 / ni) * res
    end

    # println("PosUsers: midway gradient is : $finalRes")

    for userId in posUsers
        ui = U[: , userId]
        userVec = X[userId, :]
        ni = size(userVec)[1]
        @assert (ni != 0) "PNORM:get_p_norm_gradient_by_item: ni is 0, $itemId, $userId"
        posItemIdxs = get_pos_items(userVec,relThreshold)
        negItemIdxs = get_neg_items(userVec,relThreshold)
        res = 0
        for negItemIdx in negItemIdxs
            negItemVec = V[:, negItemIdx]
            curRHeight = get_reverse_height(negItemVec, ui, V, posItemIdxs)
            @assert (curRHeight != Inf) "PNORM:get_p_norm_gradient_by_item:curRHeight is Inf"
            tempSum = 0
            for posItemIdx in posItemIdxs
                posItemVec = V[:, posItemIdx]
                t =  dot(ui, ( posItemVec -negItemVec))
                tempSum += sigma(t) * ui
                # TEST
                if any(isnan(tempSum))
                    println("PosUsers: curRHeight is : $curRHeight")
                    println("PosUsers: ui is : $ui")
                    println("PosUsers: tempSum is : $tempSum")
                    # println("PosUsers: res is : $res")
                    println(" ")
                end

                # END
            end
            res +=  1/(1+curRHeight)  * tempSum
            # TEST
            if any(isnan(res))
                println("RES is NAN")
                println("PosUsers: curRHeight is : $curRHeight")
                println("PosUsers: ui is : $ui")
                println("PosUsers: tempSum is : $tempSum")
                println("PosUsers: res is : $res")
                println(" ")
            end
            # END
        end
        finalRes -=  (1 / ni) * res
        # TEST
        if any(isnan(finalRes))
            println("PosUsers: ui is : $ui")
            println("PosUsers: res is : $res")
            println("PosUsers: ni is : $ni")
            println("PosUsers: finalres is : $finalRes")
            println(" ")
        end
        # END
    end
    # println("final gradient is $finalRes")
    return finalRes
end


# return optimized U, V
# ASSUME X is sparse
# ASSUME U, V are non-sparse
# @param Y is the validation set
function r_norm_optimizer(X, U, V, Y, learningRate; threshold=0.0001,regval=regval,
    relThreshold = 4, iterNum=200, k=5, metric = 2)
    isConverge = false
    preVal_obj = 0
    curVal_obj = 0
    userNum = size(U)[2]
    itemNum = size(V)[2]
    count = 1

    # plotting
    plotX = []
    plotY_obj = []
    plotY_train = []
    plotY_eval = []
    for it in 1:iterNum
        println("RNORM: On iteration $it")
        println("RNORM: Start user phase")
        preVal_obj = curVal_obj
        for i in 1:userNum
            userVec = X[i, :]
            ui = U[:, i]
            #### TEST
            gradient = get_r_norm_gradient_by_user(userVec, ui, V, relThreshold)
            ragVal =  regval * ui
            U[:, i] = (ui- learningRate * (gradient + ragVal))'
            if all(gradient .== 0)
                println(U[:, i])
                assert(all(U[:, i] .==0))
                @assert (ui != U[:,i]) "U[:,i] not updated! gradient is 0"
            end
            #### END
            # U[:, i] = (ui - (learningRate * get_p_norm_gradient_by_user(userVec, ui, U, V,p, relThreshold) + regval*ui))'
            @assert (ui != U[:,i]) "U[:,i] not updated!"
            @assert (any(isnan,U[:, i]) == false) "ui contains NaN"
            # println(learningRate)
            # println(regval)
            # println(temp1 ./ U[:,i])
            # println(temp1)
            # println(U[:,i])
        end
        println("RNORM: FINISHED user phase")
        # TEST
        println(any(U .== 0))
        println(any(U .> 100000))
        println(any(U .< -1e10))
        # END
        println("RNORM: Start item phase")
        for h in 1:itemNum
            itemId = h
            vh = V[: , h]
            # #### TEST
            gradient = get_r_norm_gradient_by_item(X, U, V, itemId,  relThreshold)
            ragVal =  regval * vh
            # # println(vh./gradient)
            # # println(ragVal)
            ########## CHECK Matrix op gives NaN
            # println(V[:,h])
            # println(gradient) ## gradient is NaN
            V[:, h] = (vh - learningRate *(gradient + ragVal))'
            # println("learning rate is $learningRate")
            # println(ragVal)
            # END
            # V[: , h] = vh - learningRate * (get_p_norm_gradient_by_item(X, U, V, itemId, p, relThreshold) + regval*vh)
            @assert (vh != V[:,h]) "V[:,h] not updated! being $(V[:,h])"
            @assert (any(isnan,V[:,h]) == false) "vh contains NaN being $(V[:,h])"
        end
        println("RNORM: FINISHED item phase")
        # test
        # assert(U_temp != U)
        # assert(V_temp != V)
        # assert(Y_temp == Y)
        curVal_eval = evaluate(U, V, Y, k = k,relThreshold = relThreshold, metric=metric)
        curVal_train = evaluate(U, V, X, k = k,relThreshold = relThreshold, metric=metric)
        # Test evaluate the loss instead
        curVal_obj = eval_obj(U, V, X, relThreshold)

        push!(plotY_eval, curVal_eval)
        push!(plotY_train, curVal_train)
        push!(plotY_obj, curVal_obj)

        # println("curVal is : $curVal")
        # println("preVal is : $preVal")
        if it == 1
            println("RI")
            continue
        # TEST run full iteration
        else
            # diff = (curVal - preVal) # maximizing the map_5
            diff = ( preVal_obj - curVal_obj) # minimizing the loss
            println("Diff is $diff")
            if diff <= threshold
                isConverge = true
                count = count+1
                break
            end
        end
        # # if Metric.is_converged(X, U, V, Y,threshold=0.0001)
        # if is_converged(X, U, V, Y,threshold=0.0001)
        #     isConverge = true
        #     break
        # end
        count = count+1
        println("RNORM: FINISHED iteration $it, curVal_obj is : $curVal_obj")
    end

    println("RNORM: EXITED at iteration $count, convergence is :$isConverge")
    println("RNORM: FINAL curVal_obj: $curVal_obj")

    println("PARAMS")
    println("learningRate: $learningRate, threshold: $threshold,
    regval:$regval, relThreshold:$relThreshold, iterNum:$iterNum, k:$k")
    println("END PARAMS")


    println("RNORN:final plotY_eval :$plotY_eval")
    println("RNORN:final plotY_obj :$plotY_obj")
    println("RNORN:final plotY_train :$plotY_train")
    curTime = Dates.value(now())
    return U, V, curTime, plotY_eval, plotY_train, plotY_obj
end
