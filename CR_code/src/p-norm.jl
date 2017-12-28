include("util.jl")
include("metric.jl")



#TODO optimize
function eval_obj(U, V, X, relThreshold, p)
    finalRes = 0
    for id in 1:size(X)[1]
        userVec = X[id, :]
        ui = U[:, id]
        ni = length(userVec)
        @assert (ni != 0) "PNORM:eval_obj: ni is 0"
        userRes = 0
        posItems = get_pos_items(userVec,relThreshold)
        negItems = get_neg_items(userVec, relThreshold)

        for negItem in negItems
            negItemIdx = parse(Int,split(negItem,":")[1])
            vj = V[:, negItemIdx]
            curHeight = get_height(vj, ui, V, posItems)
            userRes += curHeight^p
        end
        finalRes += (1/ni) * userRes

    end
    return finalRes
end


function get_p_norm_gradient_by_user(userVec,ui, V, p, relThreshold)
    ni = length(userVec)
    @assert (ni != 0) "PNORM:get_p_norm_gradient_by_user: ni is 0"
    posItems = get_pos_items(userVec,relThreshold)
    negItems = get_neg_items(userVec, relThreshold)
    # println("size of posItems $(length(posItems))")
    res = 0

    for negItem in negItems
        negItemIdx = parse(Int,split(negItem,":")[1])
        #TEST
        # println("size of V: $(size(V))")
        # println("negItemIdx : $negItemIdx")
        #END
        negItemVec = V[:, negItemIdx]
        curHeight = get_height(negItemVec, ui, V, posItems)
        tempSum = 0
        for posItem in posItems
            posItemIdx = parse(Int,split(posItem,":")[1])
            posItemVec = V[:, posItemIdx]
            t =  dot(ui, (posItemVec - negItemVec))
            # tempSum += sigma((ui' * (negItemVec - posItemVec))[1]) * ui
            tempSum += sigma(t) * (negItemVec - posItemVec)
            # println("g-user: dot product is : $t")
            # println("g-user: posItemVec is : $posItemVec")
        end
        res +=  curHeight ^ (p-1) * tempSum
        # println(" ")
        # println("g-user: ui is : $ui")
        # println("g-user: negItemVec is : $negItemVec")
        # println("g-user: tempSum is : $tempSum")
        # println("g-user: curHeight is : $curHeight")
        # println("g-user: cur gradient is : $res")
        # println(" ")
        # println(" ")
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

    println("PosUsers get gradient by item: midway gradient is : $finalRes")

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
                t =  dot(ui, ( posItemVec -negItemVec))
                # tempSum += sigma((ui' * (negItemVec - posItemVec))[1]) * ui
                tempSum += sigma(t) * ui
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
            res +=  curHeight ^ (p-1) * tempSum
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
        finalRes -=  (p / ni) * res
        # TEST
        if any(isnan(finalRes))
            temp = (p / ni) * res
            println(" RI : $temp")
            println("PosUsers: ui is : $ui")
            println("PosUsers: res is : $res")
            println("PosUsers: ni is : $ni")
            println("PosUsers: p is : $p")
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
# @param Y is the test set
function p_norm_optimizer(X, U, V, Y, learningRate; p = 2, convThreshold=0.0001,
    regval=0.001, relThreshold = 4, iterNum=200, k = 5, metric=2)
    # test
    println("In PNORM")
    # end
    isConverge = false
    preVal_obj = 0
    curVal_obj = 0
    userNum = size(U)[2]
    itemNum = size(V)[2]
    println("PNORM size of U $(size(U))")
    println("PNORM size of V $(size(V))")
    count = 1

    # plotting
    plotX = []
    plotY_obj = [] # eval obj on training set using updated U V
    plotY_train = [] # eval metric on training set using updated U V
    plotY_eval = [] # eval metric on testing set using updated U V

    for it in 1:iterNum
        println("Pnorm: On iteration $it")
        println("Start user phase")
        preVal_obj = curVal_obj
        for i in 1:userNum
            userVec = X[i, :]
            ui = U[:, i]
            #### TEST
            gradient = get_p_norm_gradient_by_user(userVec, ui, V,p, relThreshold)
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

            # println(" ")
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
        # assert(U_temp != U)
        # assert(V_temp != V)
        # assert(Y_temp == Y)
        curVal_eval = evaluate(U, V, Y, k = k, relThreshold = relThreshold, metric=metric)
        curVal_train = evaluate(U, V, X, k = k, relThreshold = relThreshold,metric=metric)
        # Test evaluate the loss instead
        curVal_obj = eval_obj(U, V, X, relThreshold, p)

        push!(plotY_eval, curVal_eval)
        push!(plotY_train, curVal_train)
        push!(plotY_obj, curVal_obj)
        println("curVal_eval is $curVal_eval")
        println("curVal_train is $curVal_train")
        println("curVal_obj is $curVal_obj")
        # println("curVal is : $curVal")
        # println("preVal is : $preVal")
        if it == 1
            println("RI")
            continue
        # TEST run full iteration
        # else
        #     # diff = (curVal - preVal) # maximizing the map_5
        #     diff = ( preVal_obj - curVal_obj) # minimizing the loss
        #     println("Diff is $diff")
        #     if diff <= convThreshold
        #         isConverge = true
        #         count = count+1
        #         break
        #     end
        end
        # END TEST
        count = count+1
        println("Pnorm: FINISHED iteration $it, curVal_obj is : $curVal_obj")
    end

    println("Pnorm: EXITED at iteration $count, convergence is :$isConverge")
    println("FINAL curVal_obj: $curVal_obj")
    println("PARAMS")
    println("learningRate: $learningRate, p:$p , convThreshold: $convThreshold,
    regval:$regval, relThreshold:$relThreshold, iterNum:$iterNum, k:$k")
    println("END PARAMS")
    # Plotting
    plotX = collect(1:length(plotY_obj))
    # title("minimizing loss")
    # ylabel("value of loss")
    title("PNORM maximizing map@5")
    ylabel("value of map@5")
    xlabel("iterations")
    # plot(plotX, plotY_obj, color="red", linewidth =2.0)
    # show()
    # savefig("/Users/Weilong/Desktop/temp1.png")
    plot(plotX, plotY_eval, color="blue", linewidth =2.0)
    plot(plotX, plotY_train, color = "green", linewidth =2.0)
    curTime = Dates.value(now())
    savefig("/Users/Weilong/Desktop/out_figure/pnorm_ml100k_given20_$curTime.png")
    # show()
    return U, V, curTime
end
