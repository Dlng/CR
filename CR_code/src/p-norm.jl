include("util.jl")
include("metric.jl")




# TODO TEST !!!
function get_p_norm_gradient_by_user(userVec,ui,  U, V, p, relThreshold)
    ni = size(userVec)[1]
    @assert (ni != 0) "PNORM:get_p_norm_gradient_by_user: ni is 0"
    posItems = get_pos_items(userVec,relThreshold)
    negItems = get_neg_items(userVec, relThreshold)
    println("size of posItems $(length(posItems))")
    res = 0

    for negItem in negItems
        negItemIdx = parse(Int,split(negItem,":")[1])
        negItemVec = V[:, negItemIdx]
        curHeight = get_height(negItemVec, ui, V, posItems)
        tempSum = 0
        for posItem in posItems
            posItemIdx = parse(Int,split(posItem,":")[1])
            posItemVec = V[:, posItemIdx]
            t =  dot(ui, (posItemVec - negItemVec))
            # tempSum += sigma((ui' * (negItemVec - posItemVec))[1]) * ui
            tempSum += sigma(t) * (negItemVec - posItemVec)
            println("g-user: dot product is : $t")
            println("g-user: posItemVec is : $posItemVec")
        end
        res +=  curHeight ^ (p-1) * tempSum
        println(" ")
        println("g-user: ui is : $ui")
        println("g-user: negItemVec is : $negItemVec")
        println("g-user: tempSum is : $tempSum")
        println("g-user: curHeight is : $curHeight")
        println("g-user: cur gradient is : $res")
        println(" ")
        println(" ")
        #TEST
        if all(res .== 0)

            println(size(posItems))
            println(size(negItems))

            println(curHeight)
            println(tempSum)
            println(" ")
        end
        #END
    end
    return (p / ni)  * res
end

# TODO TEST !!!
function get_p_norm_gradient_by_item(X, U, V, itemId, p,relThreshold)
    posUsers = get_pos_users(X, itemId,relThreshold)
    negUsers = get_neg_users(X, itemId,relThreshold)
    # println(size(posUsers))
    # println(size(negUsers))
    finalRes = 0
    for userId in negUsers
        ui = U[:,userId]
        userVec = X[userId, :]
        ni = size(userVec)[1]
        @assert (ni != 0) "PNORM:get_p_norm_gradient_by_item: ni is 0, $itemId, $userId"
        posItems = get_pos_items(userVec,relThreshold)
        negItems = get_neg_items(userVec, relThreshold)
        res = 0
        for negItem in negItems
            negItemIdx = parse(Int,split(negItem,":")[1])
            negItemVec = V[:, negItemIdx]
            curHeight = get_height(negItemVec, ui, V, posItems)
            @assert (curHeight != Inf) "PNORM:get_p_norm_gradient_by_item:curHeight is Inf"
            tempSum = 0
            for posItem in posItems
                posItemIdx = parse(Int,split(posItem,":")[1])
                posItemVec = V[:, posItemIdx]
                t =  dot(ui, ( posItemVec -negItemVec))
                # tempSum += sigma((ui' * (negItemVec - posItemVec))[1]) * ui
                tempSum += sigma(t) * ui
                # println("g-item: dot product is : $t")
                # println("g-item: posItemVec is : $posItemVec")
            end

            res +=  curHeight ^ (p-1) * tempSum
            # println(" ")
            # println("g-item: ui is : $ui")
            # println("g-item: negItemVec is : $negItemVec")
            # println("g-item: tempSum is : $tempSum")
            # println("g-item: curHeight is : $curHeight")
            # println("g-item: res is : $res")
            # println(" ")
            # println(" ")
        end
        finalRes += (p / ni) * res
    end

    println("PosUsers: midway gradient is : $finalRes")

    for userId in posUsers
        ui = U[: , userId]
        userVec = X[userId, :]
        ni = size(userVec)[1]
        @assert (ni != 0) "PNORM:get_p_norm_gradient_by_item: ni is 0, $itemId, $userId"
        posItems = get_pos_items(userVec,relThreshold)
        negItems = get_neg_items(userVec,relThreshold)
        res = 0
        for negItem in negItems
            negItemIdx = parse(Int,split(negItem,":")[1])
            negItemVec = V[:, negItemIdx]
            curHeight = get_height(negItemVec, ui, V, posItems)
            @assert (curHeight != Inf) "PNORM:get_p_norm_gradient_by_item:curHeight is Inf"
            tempSum = 0
            for posItem in posItems
                posItemIdx = parse(Int,split(posItem,":")[1])
                posItemVec = V[:, posItemIdx]
                tempSum += sigma((ui' * (posItemVec-negItemVec))[1]) * ui
                # TEST
                if any(isnan(tempSum))
                    println("PosUsers: curHeight is : $curHeight")
                    println("PosUsers: ui is : $ui")
                    println("PosUsers: tempSum is : $tempSum")
                    # println("PosUsers: res is : $res")
                    println(" ")
                end

                # END
            end
            res +=  curHeight ^ (p-1) .* tempSum
            # TEST
            if any(isnan(res))
                println("RES is NAN")
                println("PosUsers: curHeight is : $curHeight")
                println("PosUsers: ui is : $ui")
                println("PosUsers: tempSum is : $tempSum")
                println("PosUsers: res is : $res")
                println(" ")
            end
            # END
        end
        # finalRes -= (p / ni) * res
        old = finalRes
        finalRes -=  (p / ni) * res
        # TEST
        if any(isnan(finalRes))
            temp = (p / ni) * res
            println(" RI : $temp")
            println(old)
            println(old - temp)
            println("PosUsers: ui is : $ui")
            println("PosUsers: res is : $res")
            println("PosUsers: ni is : $ni")
            println("PosUsers: p is : $p")
            println("PosUsers: finalres is : $finalRes")
            println(" ")
        end
        # END
    end
    println("final gradient is $finalRes")
    return finalRes
end


# return optimized U, V
# ASSUME X is sparse
# ASSUME U, V are non-sparse
# @param Y is the validation set
function p_norm_optimizer(X, U, V, Y, learningRate; p = 2, threshold=0.0001,regval=regval, relThreshold = 4)
    isConverge = false
    preVal = 0
    curVal = 0
    userNum = size(U)[2]
    itemNum = size(V)[2]
    count = 1
    # test
    U_temp = deepcopy(U)
    V_temp = deepcopy(V)
    Y_temp = deepcopy(Y)
    for it in 1:200
        println("Pnorm: On iteration $it")
        println("Start user phase")
        preVal = curVal
        for i in 1:userNum
            userVec = X[i, :]
            ui = U[:, i]
            #### TEST
            # U[:, i] = U[:, i] + 1
            gradient = get_p_norm_gradient_by_user(userVec, ui, U, V,p, relThreshold)
            ragVal =  regval * ui
            # assert(size(gradient) == size(ui))
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
        println("FINISHED user phase")
        # TEST
        println(any(U .== 0))
        println(any(U .> 100000))
        println(any(U .< -1e10))
        # END
        println("Start item phase")
        for h in 1:itemNum
            itemId = h
            vh = V[: , h]
            # #### TEST
            # itemId = 6
            # vh = V[: , 6]
            # V[:, h] = V[:, h] + 1
            gradient = get_p_norm_gradient_by_item(X, U, V, itemId, p, relThreshold)
            ragVal =  regval * vh
            # # println(vh./gradient)
            # # println(ragVal)

            println(" ")
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
        println("FINISHED item phase")
        # test
        assert(U_temp != U)
        assert(V_temp != V)
        assert(Y_temp == Y)
        curVal = evaluate(U, V, Y, relThreshold = relThreshold)
        println("curVal is : $curVal")
        println("preVal is : $preVal")
        if it == 1
            println("RI")
            continue
        else
            diff = (curVal - preVal)
            println("Diff is $diff")
            if diff <= threshold
                isConverge = true
                break
            end
        end
        # # if Metric.is_converged(X, U, V, Y,threshold=0.0001)
        # if is_converged(X, U, V, Y,threshold=0.0001)
        #     isConverge = true
        #     break
        # end
        count = count+1
        println("Pnorm: FINISHED iteration $it, curVal is : $curVal")
    end

    println("Pnorm: EXITED at iteration $count, convergence is :$isConverge")
    println("FINAL cur Val: $curVal")
    return U, V
end
