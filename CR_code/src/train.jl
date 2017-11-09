include("./p-norm.jl")
include("./i-norm.jl")
include("./r-norm.jl")
include("./metric.jl")
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
    for userId  in 1:size(X)[1]
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
function train(X, U, V, Y, T;  algo=3, p=2, infGamma=10  ,regval = 1, dimW = 10,
     learningRate =0.0001, relThreshold = 4, iterNum=200, k = 5)
    assert(isnull(U) == false)
    assert(isnull(V) == false)
    X, U,Y,T= preprocessing(X, U, Y, T,relThreshold)
    ### TEST
    U_opt, V_opt = r_norm_optimizer(X, U, V, Y, learningRate, regval=regval,
    relThreshold= relThreshold, iterNum=iterNum, k = k)
    # U_opt, V_opt = p_norm_optimizer(X, U, V, Y, learningRate, p = p, regval=regval,
    # relThreshold= relThreshold, iterNum=iterNum, k = k)
    # curval = evaluate(U, V, T, relThreshold=relThreshold)
    # println(curval)
    ##############END TEST
    # if algo == 1
    #     U, V = inf_norm_optimizer(X, U, V, Y, T, infGamma = infGamma, regval = 1/lambda, learning_rate=learning_rate)
    # elseif algo == 2
    #     U, V = reverse_height_optimizer(X, U, V, Y, T)
    # else
    #     U, V = p_norm_optimizer(X, U, V, Y, T, learningRate, p = p)
    # end
    # TEST
    # evaluate(U, V, T, relThreshold=relThreshold)
    return U_opt, V_opt
    # return U, V
    #END TEST
end

#TODO use validatation set to select model params
function select_params()
end

# NOTE should maximizing the evaluation score.
# evaludate the predictions on target dataset @param Y
# @param metric = 'ap' or 'ndcg'
function evaluate(U, V, Y; metric="ap", k=5, relThreshold =4)
    @assert (any(isnan,U) == false) "U contains NaN"
    @assert (any(isnan,V) == false) "V contains NaN"
    #TODO arr of NaN still gives a value ???
    userNum = size(U)[2]
    res = 0
    for userId in 1:userNum
        testVec = Y[userId, :]
        ui = U[:, userId]
        #TEST
        # println(testVec)
        #end
        testVecIds = [parse(Int64,split(item, ":")[1]) for item in testVec  if item != ""]
        testVecScores = [parse(Int64,split(item, ":")[2]) for item in testVec  if item != ""]
        #get predictions
        preds = [dot(ui, V[:, id]) for id in testVecIds]
        predictions = [(val, id) for (id, val) in enumerate(preds)]
        temp = sort(predictions,rev = true) # sort by first element
        y_predict_idxs = [ item[2] for item in temp]
        y_predict = [testVecScores[id] for id in y_predict_idxs]
        # get metric value
        if metric == "ap"
            res += avg_precision_k(y_predict,relThreshold, k)
        else
            y_true = sort(testVecScores, rev = true)
            res += ndcg_k(y_true, y_predict, k)
        end
    end
    # TODO seems the val is not properly caled
    return res / userNum
end
