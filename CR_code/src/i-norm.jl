include("util.jl")
include("metric.jl")

# TODO ADD early stoping for INORM
# TODO ADD regVal term for INORM

##############################Solver##################################
function projsplx(y)
    s = y'
    m = length(s)
    bget = false
    sort!(s, rev=true)
    tmpsum = 0
    for i in 1:m-1
        tmpsum = tmpsum + s[i]
        tmax = (tmpsum -1) / i
        if tmax >= s(i + 1)
            bget = true
            break
        end
    end
    if !bget
        tmax =(tmpsum + s(m) -1 )/m
    end
    x=  max(y - tmax , 0)
    return x'
end

# solver for y lagrange multiplier


function solver(G, y, infGamma, innerLngRate, epochs, fu)
    y_opt = y
    for i in 1:epochs
        temp = G' * y
        y_opt -= innerLngRate * (2* dot(G, temp) - 2 * infGamma * fu)
        y_opt = projsplx(y_opt)
    end
    return y_opt
end
################################################################

################eval obj################################################
#Note compute f1(vh)
function eval_obj_by_item(posUsers, vh, X, V, U, relThreshold)
    res = 0
    for userId in posUsers
        ui = U[:, userId]
        userVec = X[userId,:]
        ni = length(userVec)
        negItemIdxs = get_neg_items(userVec, relThreshold)
        negItemVecs = [ V[:, id] for id in negItemIdxs]
        # get max over neg vecs
        maxDot = maximum([ dot(ui,v) for v in negItemVecs])
        temp = maxDot - dot(ui,vh)
        if temp > 100
            res += 1/ni * 100
        else
            res += 1/ni * log(1 + exp(temp))
        end
    end
    return res
end


function eval_obj_by_user(ui, id , V, X,relThreshold)
    userRes = 0
    userVec = X[id, :]
    posItemIdxs = get_pos_items(userVec, relThreshold)
    negItemIdxs = get_neg_items(userVec, relThreshold)
    # find the max over j and reuse in following loop
    maxJ = 0
    for negItemIdx in negItemIdxs
        vj = V[:, negItemIdx]
        temp = dot(ui, vj)
        if temp > maxJ
            maxJ = temp
        end
    end

    for posItemIdx in posItemIdxs
        vk = V[:, posItemIdx]
        temp = maxJ + dot(-ui, vk)
        if temp > 100
            userRes += temp
        else
            userRes += log(1+ exp(temp))
        end
    end
    return userRes
end

function eval_obj(U, V, X, relThreshold)
    finalRes = 0
    for id in 1:size(X)[1]
        ui = U[:, id]
        ni = length(X[id, :])
        @assert ni != 0 "INORM: eval_obj: ni is 0"
        userRes = eval_obj_by_user(ui,id, V, X, relThreshold)
        finalRes += 1/ni * userRes
    end
    return finalRes
end



# df_j(u)/ du
function get_mini_gradient_by_user(ui,posItemVecs, negItemVec)
    res = 0
    for posVec in posItemVecs
        a_kj = posVec - negItemVec
        res += 1/ (1 + exp(dot(ui, a_kj))) * (-a_kj)
    end
    return res
end

## df^{+}_{j}(u)/ dv_h
function get_mini_gradient_by_item(ui, vh, vj)
    temp = dot(ui, vh - vj)
    return 1/(exp(temp) + 1) * -ui
end
################################################################

function get_i_norm_gradient_by_user(userVec, ui, V, relThreshold, infGamma,
    epochs, innerLngRate, fu)
    #solve for u_opt
    ni = length(userVec)

    posItemIdxs = get_pos_items(userVec,relThreshold)
    negItemIdxs = get_neg_items(userVec, relThreshold)
    negItemNum = length(negItemIdxs)
    @assert (ni != 0) "I NORM:get_i_norm_gradient_by_user: ni is 0"
    @assert (negItemNum != 0) "I NORM:get_i_norm_gradient_by_user: negItemNum is 0"
    # init y
    # y = randn(negItemNum)
    y = fill(1/negItemNum, negItemNum)
    y = y'  # y is a column
    # init G dim = n x n
    G = []
    posItemVecs = [V[:,idx] for idx in posItemIdxs]

    for negItemIdx in negItemIdxs
        negItemVec = V[:, negItemIdx]
        miniGradient = get_mini_gradient_by_user(ui,posItemVecs, negItemVec)
        push!(G,miniGradient)
    end
    @assert (size(G) == (negItemNum, negItemNum)) "I NORM:get_i_norm_gradient_by_user: wrong dim G  with $(size(G))"

    #solve for y

    opt_y = solver(G, y, infGamma, innerLngRate, epochs, fu)
    # solve for u_opt
    u_opt = ui - 1/infGamma* (dot(G, opt_y))
    return infGamma * (ui - u_opt)
end

function get_i_norm_gradient_by_item(X, U, V, itemId, relThreshold, infGamma,
     epochs, innerLngRate)
    finalGradient = 0
    posUsers = get_pos_users(X, itemId,relThreshold)
    negUsers = get_neg_users(X, itemId,relThreshold)
    vh = V[:,itemId]

    # eval f_1(vh)
    f_vh = eval_obj_by_item(posUsers, vh, X, V, U, relThreshold)
    # for posUsers follow the get_i_norm_gradient_by_user procedure

    for posUserId in posUsers #TODO check correctness, run solver so many times
        ui = U[:,posUserId]
        userVec = X[posUserId, :]
        ni = length(userVec)
        negItemIdxs = get_neg_items(userVec, relThreshold)
        negItemNum = length(negItemIdxs)
        ## init column y
        y = fill(1/negItemNum, negItemNum)
        y = y'
        ## init G
        G = []
        for negItemIdx in negItemIdxs
            vj = V[:, negItemIdx]
            temp = get_mini_gradient_by_item(ui, vh, vj)
            push!(G, temp)
        end
        @assert (size(G) == (negItemNum, negItemNum)) "I NORM:get_i_norm_gradient_by_item: wrong dim G  with $(size(G))"

        #solve for y
        opt_y = solver(G, y, infGamma, innerLngRate, epochs, f_vh)

        # solve for v_opt
        v_opt = vh - 1/infGamma * (dot(G , opt_y))
        finalGradient += 1/ni * v_opt
    end
    # for negUsers
    res = 0
    for negUserId in negUsers
        ui = U[:,negUserId]
        userVec = X[negUserId, :]
        ni = length(userVec)
        posItemIdxs = get_pos_items(userVec, relThreshold)
        posItemVecs = [ V[:, id] for id in posItemIdxs]
        posItemNum = length(posItemIdxs)
        temp = 0
        for vk in posItemVecs
            temp += 1/(1+ exp(dot(ui, vk - vh))) * ui
        end
        res += 1/ni * temp
    end
    finalGradient += res
    return finalGradient
end


function i_norm_optimizer(X, U, V, Y; learningRate=0.01, regval=0.1, infGamma=10,innerLngRate = 0.001,
convThreshold=0.0001, relThreshold= 4, iterNum=200, epochs = 200, k = 5, metric=2)
    println("In I NORM")
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
        println("I Norm: On iteration $it")
        println("Start user phase")
        preVal_obj = curVal_obj

        for i in 1:userNum
            userVec = X[i, :]
            ui = U[:, i]
            fu =  eval_obj_by_user(ui, i, V, X, relThreshold)
            gradient = get_i_norm_gradient_by_user(userVec, ui, V, relThreshold, infGamma, epochs,innerLngRate, fu)
            ragVal =  regval * ui
            U[:, i] = (ui- learningRate * (gradient + ragVal))'
            if all(gradient .== 0)
                println(U[:, i])
                assert(all(U[:, i] .==0))
                @assert (ui != U[:,i]) "U[:,i] not updated! gradient is 0"
            end
            @assert (ui != U[:,i]) "U[:,i] not updated!"
            @assert (any(isnan,U[:, i]) == false) "ui contains NaN"
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
            gradient = get_i_norm_gradient_by_item(X, U, V, itemId, relThreshold,epochs, innerLngRate)
            ragVal =  regval * vh
            V[:, h] = (vh - learningRate *(gradient + ragVal))'
            @assert (vh != V[:,h]) "V[:,h] not updated! being $(V[:,h])"
            @assert (any(isnan,V[:,h]) == false) "vh contains NaN being $(V[:,h])"
        end
        println("FINISHED item phase")
        curVal_eval = evaluate(U, V, Y, k = k, relThreshold = relThreshold, metric=metric)
        curVal_train = evaluate(U, V, X, k = k, relThreshold = relThreshold,metric=metric)
        # Test evaluate the loss instead
        curVal_obj = eval_obj(U, V, X, relThreshold, p)

        push!(plotY_eval, curVal_eval)
        push!(plotY_train, curVal_train)
        push!(plotY_obj, curVal_obj)

        if it == 1
            println("RI")
            continue
        else
            # diff = (curVal - preVal) # maximizing the map_5
            diff = ( preVal_obj - curVal_obj) # minimizing the loss
            println("Diff is $diff")
            if diff <= convThreshold
                isConverge = true
                count = count+1
                break
            end
        end
        count = count+1
        println("I Norm: FINISHED iteration $it, curVal_obj is : $curVal_obj")
    end

    println("I Norm: EXITED at iteration $count, convergence is :$isConverge")
    println("FINAL curVal_obj: $curVal_obj")
    println("PARAMS")
    println("learningRate: $learningRate, convThreshold: $convThreshold,regval:$regval,
    relThreshold:$relThreshold, iterNum:$iterNum, k:$k, infGamma:$infGamma,
     innerLngRate:$innerLngRate, convThreshold: $convThreshold,  epochs:$epochs ")
    println("END PARAMS")
    curTime = Dates.value(now())
    return U, V, curTime, plotY_eval, plotY_train, plotY_obj
end
