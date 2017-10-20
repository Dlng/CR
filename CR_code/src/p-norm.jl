include("util.jl")
include("metric.jl")




# TODO TEST !!!
function get_p_norm_gradient_by_user(userVec,ui,  U, V, p)
    ni = size(userVec)[1]
    @assert (ni != 0) "PNORM:get_p_norm_gradient_by_user: ni is 0"
    posItems = get_pos_items(userVec)
    negItems = get_neg_items(userVec)
    res = 0

    for negItem in negItems
        negItemIdx = parse(Int,split(negItem,":")[1])
        negItemVec = V[:, negItemIdx]
        curHeight = get_height(negItemVec, ui, V, posItems)
        tempSum = 0
        for posItem in posItems
            posItemIdx = parse(Int,split(posItem,":")[1])
            posItemVec = V[:, posItemIdx]
            tempSum += sigma((ui' * (negItemVec - posItemVec))[1]) *
             (negItemVec - posItemVec)
        end
        res +=  curHeight ^ (p-1) .* tempSum
    end

    return (p / ni)  * res
end

# TODO TEST !!!
function get_p_norm_gradient_by_item(X, U, V, itemId, p)
    posUsers = get_pos_users(X, itemId)
    negUsers = get_neg_users(X, itemId)

    finalRes = 0
    for userId in posUsers
        ui = U[:,userId]
        userVec = X[userId, :]
        ni = size(userVec)[1]
        @assert (ni != 0) "PNORM:get_p_norm_gradient_by_item: ni is 0, $itemId, $userId"
        posItems = get_pos_items(userVec)
        negItems = get_neg_items(userVec)
        res = 0
        for negItem in negItems
            negItemIdx = parse(Int,split(negItem,":")[1])
            negItemVec = V[:, negItemIdx]
            curHeight = get_height(negItemVec, ui, V, posItems)
            println("PosUsers: curHeight is : $curHeight")
            ###TEST
            if curHeight == Inf
                println(negItemVec)
                println(ui)
                println(posItems)
                println(any(isnan(V)))
            end
            ### END TEST
            tempSum = 0
            for posItem in posItems
                posItemIdx = parse(Int,split(posItem,":")[1])
                posItemVec = V[:, posItemIdx]
                tempSum += sigma((ui' * (negItemVec - posItemVec))[1]) * ui
            end

            println("PosUsers: ui is : $ui")
            println("PosUsers: tempSum is : $tempSum")
            res +=  curHeight ^ (p-1) .* tempSum
        end
        finalRes += p / ni * res

    end
    println(finalRes)
    for userId in negUsers
        ui = U[: , userId]
        userVec = X[userId, :]
        ni = size(userVec)[1]
        @assert (ni != 0) "PNORM:get_p_norm_gradient_by_item: ni is 0, $itemId, $userId"
        posItems = get_pos_items(userVec)
        negItems = get_neg_items(userVec)
        res = 0
        for negItem in negItems
            negItemIdx = parse(Int,split(negItem,":")[1])
            negItemVec = V[:, negItemIdx]
            curHeight = get_height(negItemVec, ui, V, posItems)
            tempSum = 0
            for posItem in posItems
                posItemIdx = parse(Int,split(posItem,":")[1])
                posItemVec = V[:, posItemIdx]
                tempSum += sigma((ui' * (negItemVec - posItemVec))[1]) * ui
            end
            res +=  curHeight ^ (p-1) .* tempSum
        end
        finalRes -= p / ni * res
    end
    return finalRes
end


# return optimized U, V
# ASSUME X is sparse
# ASSUME U, V are non-sparse
# @param Y is the validation set
function p_norm_optimizer(X, U, V, Y, learningRate; p = 2, threshold=0.0001,regval=regval)
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
        #TEST
        temp1 =[]
        temp2 =[]
        #END
        for i in 1:userNum
            userVec = X[i, :]
            ui = U[:, i]
            #### TEST
            U[:, i] = U[:, i] + 1
            gradient = learningRate * get_p_norm_gradient_by_user(userVec, ui, U, V,p)
            ragVal =  regval * ui
            # # println(ragVal./)
            # println(ragVal./ui)
            # println(gradient./ui)
            # println(gradient)
            # println(typeof(ui))
            # println(typeof(U[:, i]))
            # println(typeof(gradient))
            # println(typeof(ui-gradient + ui))
            # println(typeof(ui-gradient + ui)')
            temp1 = U[:, i]
            # assert(size(gradient) == size(ui))
            U[:, i] = (ui- gradient + ragVal)'
            #### END
            # U[:, i] = ui - learningRate * get_p_norm_gradient_by_user(userVec, ui, U, V,p) + regval*ui
            @assert (ui != U[:,i]) "U[:,i] not updated!"
            @assert (any(isnan,U[:, i]) == false) "ui contains NaN"
            # println(learningRate)
            # println(regval)
            # println(temp1 ./ U[:,i])
            # println(temp1)
            # println(U[:,i])
        end
        println("FINISHED user phase")
        println("Start item phase")
        for h in 1:itemNum
            itemId = h
            vh = V[: , h]
            # #### TEST
            # itemId = 6
            # vh = V[: , 6]
            # V[:, h] = V[:, h] + 1
            gradient = learningRate * get_p_norm_gradient_by_item(X, U, V, itemId, p)
            ragVal =  regval * vh
            # # println(vh./gradient)
            # # println(ragVal)
            # temp2 = gradient ./ vh
            # println(ragVal./vh)
            # println(gradient./vh)
            # println(gradient)
            ########## CHECK Matrix op gives NaN
            println(typeof(vh))
            println(typeof(V[:, h]))
            println(typeof(gradient))
            println(typeof(vh-gradient + vh))
            println(typeof((vh-gradient + vh)'))
            # println(V[:,h])
            # println(gradient) ## gradient is NaN
            V[:, h] = (vh - gradient + ragVal)'
            # println(ragVal)
            # END
            # V[: , h] = vh - learningRate * get_p_norm_gradient_by_item(X, U, V, itemId, p) + regval*vh
            @assert (vh != V[:,h]) "V[:,h] not updated! being $(V[:,h])"
            @assert (any(isnan,V[:,h]) == false) "vh contains NaN being $(V[:,h])"
        end
        println("FINISHED item phase")
        # TEST
        # temp = temp1 .* temp2
        # println("TEST: $temp")
        # END
        # test
        assert(U_temp != U)
        assert(V_temp != V)
        assert(Y_temp == Y)
        curVal = evaluate(U, V, Y)
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
