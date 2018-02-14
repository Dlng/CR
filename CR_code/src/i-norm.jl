include("util.jl")
include("metric.jl")

# TODO ADD early stoping for INORM
# TODO ADD regVal term for INORM

##############################Solver##################################
function projsplx(y)
    #TEST
    # debug("in projsplx $y")
    # debug(size(y))
    # debug(vec(y))
    # debug(size(vec(y)))
    # debug(" ")
    #END
    s = deepcopy(vec(y))
    m = length(s)
    bget = false
    sort!(s, rev=true)
    tmpsum = 0
    for i in 1:m-1
        tmpsum = tmpsum + s[i]
        tmax = (tmpsum -1) / i
        if tmax >= s[i + 1]
            bget = true
            break
        end
    end
    if !bget
        tmax =(tmpsum + s[m] -1 )/m
    end
    x=  max.(y - tmax , 0)
    return x'
end

# solver for y lagrange multiplier


function solver(G, y, infGamma, innerLngRate, epochs, fu)
    y_opt = y
    dim_y = size(y)
    for i in 1:epochs
        temp = G' * y #dimW x negItemNum  x negItemNum x 1 = dimW x 1
        y_opt -= innerLngRate * (2* G * temp - 2 * infGamma * fu)
        # @assert(size(y_opt) == (negItemNum,1))
        proj =  projsplx(y_opt)
        y_opt = reshape(proj,dim_y)
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

        negItemIdxs = get_neg_items(userVec, relThreshold)
        posItemIdxs = get_pos_items(userVec, relThreshold)
        ni = length(posItemIdxs) + length(negItemIdxs)
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

function eval_obj(U, V, X, relThreshold, regval)
    finalRes = 0
    for id in 1:size(X)[1]
        ui = U[:, id]
        userVec = X[id, :]
        posItemIdxs = get_pos_items(userVec, relThreshold)
        negItemIdxs = get_neg_items(userVec, relThreshold)
        ni = length(posItemIdxs) + length(negItemIdxs)
        @assert ni != 0 "INORM: eval_obj: ni is 0"
        userRes = eval_obj_by_user(ui,id, V, X, relThreshold)
        finalRes += 1/ni * userRes
    end
    # add reg terms
    regTerm = regval/2 * (dot(U,U) ^2 + dot(V,V) ^2)
    finalRes += regTerm
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

function get_i_norm_gradient_by_user(userVec, ui, V, relThreshold, infGamma,epochs, innerLngRate, fu, innerConvThreshold)
    #solve for u_opt
    posItemIdxs = get_pos_items(userVec,relThreshold)
    negItemIdxs = get_neg_items(userVec, relThreshold)
    negItemNum = length(negItemIdxs)
    posItemNum = length(posItemIdxs)
    ni =  posItemNum + negItemNum
    @assert (ni != 0) "I NORM:get_i_norm_gradient_by_user: ni is 0"
    @assert (negItemNum != 0) "I NORM:get_i_norm_gradient_by_user: negItemNum is 0"
    # init y negItemNum x 1
    y = fill(1/negItemNum, (negItemNum,1))
    # init G dim = negItemNum x dimW
    dimW = size(V)[1]

    # init G
    G = zeros(negItemNum,dimW)
    posItemVecs = [V[:,idx] for idx in posItemIdxs]

    G_row_id = 1
    for negItemIdx in negItemIdxs
        negItemVec = V[:, negItemIdx]
        miniGradient = get_mini_gradient_by_user(ui,posItemVecs, negItemVec)
        G[G_row_id, :] = miniGradient'
        G_row_id += 1
    end
    @assert (size(G) == (negItemNum,dimW)) "I NORM:get_i_norm_gradient_by_user: wrong dim G  with $(size(G))"

    #solve for y

    opt_y = solver(G, y, infGamma, innerLngRate, epochs, fu)
    # solve for u_opt
    u_opt = ui - 1/infGamma* (G' * opt_y)
    return infGamma * (ui - u_opt)
end

function get_i_norm_gradient_by_item(X, U, V, itemId, relThreshold, infGamma,
     epochs, innerLngRate, innerConvThreshold)
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
        negItemIdxs = get_neg_items(userVec, relThreshold)
        posItemIdxs = get_pos_items(userVec, relThreshold)
        ni = length(posItemIdxs) + length(negItemIdxs)
        negItemNum = length(negItemIdxs)
        ## init column y
        y = fill(1/negItemNum, (negItemNum,1))
        ## init G
        dimW = size(V)[1]
        G_row_id = 1
        G = zeros(negItemNum, dimW)
        for negItemIdx in negItemIdxs
            vj = V[:, negItemIdx]
            temp = get_mini_gradient_by_item(ui, vh, vj)
            G[G_row_id, :] = temp'
            G_row_id += 1
        end
        @assert (size(G) == (negItemNum,dimW)) "I NORM:get_i_norm_gradient_by_item: wrong dim G  with $(size(G))"

        #solve for y
        opt_y = solver(G, y, infGamma, innerLngRate, epochs, f_vh)

        # solve for v_opt
        v_opt = vh - 1/infGamma * (G' * opt_y)
        finalGradient += 1/ni * v_opt
    end
    # for negUsers
    res = 0
    for negUserId in negUsers
        ui = U[:,negUserId]
        userVec = X[negUserId, :]

        posItemIdxs = get_pos_items(userVec, relThreshold)
        negItemIdxs = get_neg_items(userVec, relThreshold)
        ni = length(negItemIdxs) + length(posItemIdxs)
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


function i_norm_optimizer(X, U, V, Y, T; learningRate=0.01, regval=0.1, infGamma=10,innerLngRate = 0.001, innerConvThreshold=0.01,
convThreshold=0.0001, relThreshold= 4, iterNum=200, epochs = 200, k = 5, metric=2)
    debug("In I NORM")
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
    plotY_vali = []

    for it in 1:iterNum
        debug("I Norm: On iteration $it")
        debug("Start user phase")
        preEvalVali = curEvalVali
        for i in 1:userNum
            userVec = X[i, :]
            ui = U[:, i]
            fu =  eval_obj_by_user(ui, i, V, X, relThreshold)
            gradient = get_i_norm_gradient_by_user(userVec, ui, V, relThreshold,
             infGamma, epochs,innerLngRate, fu, innerConvThreshold)
            ragVal =  regval * ui
            # U[:, i] = (ui- learningRate * (gradient + ragVal))' # the transpose is now unnecessary in v0.6.2 ?
            U[:, i] = ui- learningRate * (gradient + ragVal)
            if all(gradient .== 0)
                debug(U[:, i])
                assert(all(U[:, i] .==0))
                @assert (ui != U[:,i]) "U[:,i] not updated! gradient is 0"
            end
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
            itemId = h
            vh = V[: , h]

            gradient = get_i_norm_gradient_by_item(X, U, V, itemId, relThreshold,
             infGamma,epochs, innerLngRate,innerConvThreshold)
            ragVal =  regval * vh
            V[:, h] = vh - learningRate *(gradient + ragVal)
            @assert (vh != V[:,h]) "V[:,h] not updated! being $(V[:,h])"
            @assert (any(isnan.(V[:,h])) == false) "vh contains NaN being $(V[:,h])"
        end
        debug("FINISHED item phase")
        curEvalVali = evaluate(U, V, Y, k = 5, relThreshold = relThreshold, metric=1) # using MAP@5
        curEvalTest = evaluate(U, V, T, k = k, relThreshold = relThreshold, metric=metric)
        curEvalTrain = evaluate(U, V, X, k = k, relThreshold = relThreshold,metric=metric)
        # Test evaluate the loss instead
        curVal_obj = eval_obj(U, V, X, relThreshold,regval)

        push!(plotY_eval, curEvalTest)
        push!(plotY_train, curEvalTrain)
        push!(plotY_vali, curEvalVali)
        push!(plotY_obj, curVal_obj)
        debug("curEvalTest is $curEvalTest")
        debug("curEvalVali is $curEvalVali")
        debug("curEvalTrain is $curEvalTrain")
        debug("curVal_obj is $curVal_obj")

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
        #TEST
        # if count == iterNum
        #     break
        # end
        #END
        count = count+1
        debug("I Norm: FINISHED iteration $it, curVal_obj is : $curVal_obj")
    end

    debug("I Norm: EXITED at iteration $count, convergence is :$isConverge")
    debug("INORN:final plotY_eval :$plotY_eval")
    debug("INORN:final plotY_obj :$plotY_obj")
    debug("INORN:final plotY_train :$plotY_train")
    debug("INORN:final plotY_vali :$plotY_vali")

    debug("INORM FINISH")

    return U, V,  plotY_eval, plotY_train, plotY_obj
end
