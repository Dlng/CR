using MATLAB

include("util.jl")
include("solve_trace_reg.jl")

#TODO do param selection on regval 1e−4, 1e−3, 1e−2, 1e−1,1


function eval_obj(U, V , X, params)
    relThreshold = params["relThreshold"]
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


function get_convex_r_norm_gradient(U,V,X,params)
"""
- since M should not be computed explicitly, for each gradient entry, coord should be recorded
- How to reflect the changes in U, V? or what does solve_trace_reg expect?
""" 
    relThreshold = params["relThreshold"]
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


function convex_r_norm_optimizer(X, M, Y, T, params; convThreshold=0.0001,
    regval=0.001, relThreshold = 4, max_iter=10, k = 5, metric=2)
    
    # make args
    m = size(U,2)
    n = size(V,2)
    dims = [m,n]
    lambda = 10

    opts = Dict()
    lbfgsb_in = Dict()

    #init lbfgsb_in
    lbfgsb_in["maxIter"] = 5;         #% max number of iterations
    lbfgsb_in["maxFnCall"] = 1000;    #% max number of calling the function
    lbfgsb_in["relCha"] = 1e-3;   #% tolerance of constraint satisfaction
    lbfgsb_in["tolPG"] = 1e-3;       # % final objective function accuracy parameter
    lbfgsb_in["m"] = 20
    #init opts
    lbfgsb_in["maxFnCall"] = 1000; # % max number of calling the function
    opts["rtol"] = 1e-3;       #% terminate if relative difference falls below it

    opts["lbfgsb_in"] = lbfgsb_in
    opts["init_rank"] = 2
    opts["max_iter"] = max_iter # the number of rank, 10 is the val in CR with push    
    opts["use_local"] = true;  #% whether to use local optimization at each iteration
    opts["verbose"] = true
    opts["lambda"] =lambda
    # params for eval_obj & eval_gradient
    opts["relThreshold"] = relThreshold
    opts["k"] =k
    opts["metric"] =metric
    # get solution from gcg
    debug("convex_r_norm_optimizer, START gcg")
    U_opt, V_opt, plotY_eval, plotY_train, plotY_obj = solve_trace_reg(X,Y,T,eval_obj,eval_gradient,evalf, opts)
    debug("convex_r_norm_optimizer, END gcg")
    return U_opt, V_opt , plotY_eval, plotY_train,plotY_obj
end
