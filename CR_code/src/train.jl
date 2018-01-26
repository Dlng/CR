include("p-norm.jl")
include("i-norm.jl")
include("r-norm.jl")
include("metric.jl")
# relevent thresholds
## >=4 : Movielens, Yahoo
## >=5: EachMovie

# additonal filtering require: remove user with no pos rating for regular height, or with no neg rating for reverse height
TRAIN_PATH = "/Users/Weilong/Desktop/Webscope_R1/train.lsvm"
VALIDATE_PATH = "/Users/Weilong/Desktop/Webscope_R1/validate.lsvm"
UPATH = "/Users/Weilong/Codes/cofirank/out_bak/U.lsvm"
VPATH = "/Users/Weilong/Codes/cofirank/out_bak/M.lsvm"

# remove users with no neg or pos rating, remove the matching cols in U,V too
function preprocessing(X, U, Y, T, relThreshold)
    idxs = []
    numRow = size(X)[1]
    for userId  in 1:numRow
        pos_counter = 0
        neg_counter = 0
        row = X[userId,:]
        for item in row
            if parse(Int,split(item,":")[2]) >= relThreshold
                pos_counter += 1
            else
                neg_counter += 1
            end
        end
        if pos_counter == 0 || neg_counter == 0
            push!(idxs, userId)
        end
    end
    X = X[setdiff(1:end, idxs), :] # m,ni
    U = U[:, setdiff(1:end, idxs)] # d,m
    Y = Y[setdiff(1:end, idxs), : ] # m,10
    T = T[setdiff(1:end, idxs), :] # m,20 (or more)
    return X, U, Y, T
end

# @param infGamma is for inf push,
# @param regval is the regularization coeff.
function train(X, U, V, Y, T, plotDir, dataset, useCofi, curTime, ni;  algo=2, p=2, infGamma=10  ,regval = 1,convThreshold=0.0001,
     learningRate =0.0001, relThreshold = 4, iterNum=200, k = 5, metric=2, epochs=200, innerLngRate=0.01, innerConvThreshold=0.01)
    assert(isnull(U) == false)
    assert(isnull(V) == false)
    X, U,Y,T= preprocessing(X, U, Y, T,relThreshold)
    debug("TEST: $algo")
    debug("TEST: $typeof(algo)")
    tic() # for timing
    if algo == 1
        U_opt, V_opt, plotY_eval, plotY_train,plotY_obj =
        r_norm_optimizer(X, U, V, Y, T, learningRate=learningRate,
        convThreshold=convThreshold, regval=regval, relThreshold= relThreshold,
        iterNum=iterNum, k = k, metric=metric)
    elseif algo == 2
        U_opt, V_opt , plotY_eval, plotY_train,plotY_obj =
         p_norm_optimizer(X, U, V, Y, T, learningRate, p = p,convThreshold=convThreshold, regval=regval,
        relThreshold= relThreshold, iterNum=iterNum, k = k, metric=metric)
    else
        regval = 1/infGamma
        U_opt, V_opt, plotY_eval, plotY_train,plotY_obj =
        i_norm_optimizer(X, U, V, Y, T, learningRate=learningRate, regval=regval,
        infGamma=infGamma,innerLngRate = innerLngRate,innerConvThreshold=innerConvThreshold, convThreshold=convThreshold,
        relThreshold= relThreshold, iterNum=iterNum,epochs = epochs, k = k, metric=metric)
    end

    toc() # for timing
    plotFigure(plotDir, curTime, dataset, useCofi,algo, metric, ni, k, plotY_eval, plotY_train, plotY_obj)

    return U_opt, V_opt
end

#TODO use validatation set to select regVal
function select_params()
end

# NOTE should maximizing the evaluation score.
# evaludate the predictions on target dataset @param Y
# @param metric = 'ap' or 'ndcg'
function evaluate(U, V, Y; k=5, relThreshold =4, metric=1)
    @assert (any(isnan,U) == false) "U contains NaN"
    @assert (any(isnan,V) == false) "V contains NaN"
    userNum = size(U)[2]
    res = 0
    for userId in 1:userNum
        testVec = Y[userId, :]
        ui = U[:, userId]
        testVecIds = [parse(Int64,split(item, ":")[1]) for item in testVec  if item != ""]
        testVecScores = [parse(Int64,split(item, ":")[2]) for item in testVec  if item != ""]
        #get predictions
        preds = [dot(ui, V[:, id]) for id in testVecIds] # TODO: BUG idx out of bound
        predictions = [(val, id) for (id, val) in enumerate(preds)]
        temp = sort(predictions,rev = true) # sort by first element
        y_predict_idxs = [ item[2] for item in temp]
        y_predict = [testVecScores[id] for id in y_predict_idxs]
        # get metric value
        if metric == 1
            res += avg_precision_k(y_predict,relThreshold, k)
        else
            y_true = sort(testVecScores, rev = true)
            res += ndcg_k(y_true, y_predict, k)
        end
    end
    return res / userNum
end
