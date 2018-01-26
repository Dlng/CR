

"""
Util
"""
function read_numeric_matrix_from_file(path,m,dimW)
    # debug("m is $m")
    # debug("dimW is $dimW")
    res = zeros(m,dimW)
    @assert all(res .== 0)
    rowCnt = 1
    open(path, "r") do f
        for line in eachline(f)
            # debug(typeof(line))
            # debug(line)
            # debug(isempty(line))
            # debug(length(line))
            # if line == ""
            if length(line) == 1 || line == ""
                res[rowCnt, :] = randn(dimW)
            else
                tempLine = split(strip(line),' ')
                tempLine2 = [parse(Float64, x) for x in tempLine] #TODO error parse("")
                #TEST
                if any(tempLine2 .>= 1e5)
                    print(tempLine2)
                end
                #END
                res[rowCnt, :] = tempLine2
            end
            rowCnt += 1
        end
    end

    return res
end

function write_to_file(path, matrix)
    open(path, "w") do f
        for rowNum in 1:size(matrix)[1]
            row = matrix[rowNum, :]
            write(f, "$row \n")
        end
    end
end

# TODO select a mini batch of users instead
# ASSUME X is sparse
# return the list of ids of users that rank cur item relevant
# @param X: the raw input matrix
# @param itemId: the col idx in V
function get_pos_users(X, itemId, relThreshold)
    res = []
    for userId in  1:size(X)[1]
        row = X[userId,:]
        for item in row
            temp = split(item, ":")
            if itemId == parse(Int,temp[1])
                if parse(Int,temp[2]) >= relThreshold
                    push!(res,userId)
                end
                break
            end
        end
    end
    return res
end


#ASSUME X is sparse
# return the list of ids of users that rank cur item irrelevant
# @param X: the raw input matrix
# @param itemId: the col idx in V
function get_neg_users(X, itemId,  relThreshold)
    res = []
    for userId in  1:size(X)[1]
        row = X[userId,:]
        for item in row
            temp = split(item, ":")
            if itemId == parse(Int,temp[1])
                if parse(Int,temp[2]) < relThreshold
                    push!(res,userId)
                end
                break
            end
        end
    end
    return res
end

# get relevent items for user i
# @param userVec: vector of strings in forms of "itemId:rating"

function get_pos_items(userVec, relThreshold)
    res = []
    for item in userVec
        if item == ""
            continue
        end
        temp = split(item,":")
        itemVal = parse(Int,temp[2])
        itemIdx = parse(Int,temp[1])
        if itemVal >= relThreshold
            push!(res, itemIdx)
        end
    end
    return res
end


# get non-relevent items for user i
function get_neg_items(userVec, relThreshold)
    res = []
    for item in userVec
        if item == ""
            continue
        end
        temp = split(item,":")
        itemVal = parse(Int,temp[2])
        itemIdx = parse(Int,temp[1])
        if itemVal < relThreshold
            push!(res, itemIdx)
        end
    end
    return res
end

function get_height_convex(userRowM, curNegItemVal, posItemIdxs)
    curHeight = 0
    for posItemIdx in posItemIdxs
        curPosItemVal = userRow[posItemIdx]
        delta = curPosItemVal - curNegItemVal
        ri = 0
        if delta > -100  # FIX
            @assert (exp(-delta) != Inf) "exp(-delta) is Inf"
            ri = log(1 + exp(-delta))
        else
            ri = -delta
        end
        @assert (!isnan(delta)) "delta is nan in get height"
        #END TEST
        curHeight += ri
    end
    @assert (curHeight != Inf) "curHeight is Inf"
    @assert (isnan(curHeight) == false) "curHeight is NaN"
    @assert (curHeight >= 0) "curHeight is $curHeight"
    return curHeight
end

# @param following convention from paper, xj is V[:,j]
function get_height(xj, ui, V, posItemIdxs)
    # debug("get_height xj: $xj")
    # debug("get_height ui: $ui")
    curHeight = 0
    for posItemIdx in posItemIdxs
        posItemVec = deepcopy(V[:, posItemIdx])
        delta = dot(ui, (posItemVec - xj))
        ri = 0
        # debug("delta is $delta")
        if delta > -100  # FIX
            @assert (exp(-delta) != Inf) "exp(-delta) is Inf"
            ri = log(1 + exp(-delta))
        else
            ri = -delta
        end
        @assert (!isnan(delta)) "delta is nan in get height"
        #TEST
        if ri == Inf || isnan(ri) || isnan(curHeight)
            temp = posItemVec-xj
            temp2 = ui' * temp # why this is NAN
            debug("in get height ")
            debug(ri)
            debug(temp)
            debug(temp2)
            debug(delta)

            debug(size(ui) == size(temp))

            debug(ui)
            debug(posItemVec)
            debug(xj)
            debug(" ")
            # debug(log((1 + exp(-delta))))
        end
        #END TEST
        curHeight += ri

    end
    @assert (curHeight != Inf) "curHeight is Inf"
    @assert (isnan(curHeight) == false) "curHeight is NaN"
    @assert (curHeight >= 0) "curHeight is $curHeight"
    return curHeight
end


# return the vector of heights of irrelevent item for user i
# @ param: userVec: the row of user in X
# @ param: ui, the row of user in U
function get_heights(userVec, ui, V)
    res = []
    posItems = get_pos_items(userVec)
    negItems = get_neg_items(userVec)

    for negItem in negItems
        negItemIdx = parse(Int, split(negItem, ":")[1])
        negItemVec = V[: , negItemIdx]
        curHeight = get_height(negItemVec, ui, V, posItems)
        push!(res, curHeight)
    end
    return res
end


function get_reverse_height_convex(userRowM, curPosItemVal, negItemIdxs)
    curRHeight = 0
    for negItemIdx in negItemIdxs
        curNegItemVal = userRow[negItemIdx]
        delta = curPosItemVal - curNegItemVal
        ri = 0
        if delta > -100  # FIX
            @assert (exp(-delta) != Inf) "exp(-delta) is Inf"
            ri = log(1 + exp(-delta))
        else
            ri = -delta
        end
        @assert (!isnan(delta)) "delta is nan in get height"
        curRHeight += ri
    end
    @assert (curRHeight != Inf) "curRHeight is Inf"
    @assert (isnan(curRHeight) == false) "curRHeight is NaN"
    @assert (curRHeight >= 0) "curRHeight is $curRHeight"
    return curRHeight
end

function get_reverse_height(xk, ui, V, negItemIdxs)
    curRHeight = 0
    for negItemIdx in negItemIdxs
        negItemVec = V[:, negItemIdx]
        delta = dot(ui, (xk - negItemVec))
        ri = 0
        # debug("delta is $delta")
        if delta > -100 # FIX
            @assert (exp(-delta) != Inf) "exp(-delta) is Inf"
            ri = log(1 + exp(-delta))
        else
            ri = -delta
        end
        curRHeight += ri

    end
    @assert (curRHeight != Inf) "curRHeight is Inf"
    @assert (isnan(curRHeight) == false) "curRHeight is NaN"
    @assert (curRHeight >= 0) "curRHeight is $curRHeight"
    return curRHeight
end

#
#return the vector of reverse heights of relevent item for user i
# function get_reverse_heights(userVec, ui, V)
#     res = []
#     posItems = get_pos_items(userVec)
#     negItems = get_neg_items(userVec)
#
#     for posItem in posItems
#         posItemIdx = parse(Int,split(posItem, ":")[1])
#         posItemVec = V[: , posItemIdx]
#         curHeight = 0.0
#         for negItem in negItems
#             negItemIdx = parse(Int,split(negItem,":")[1])
#             negItemVec = V[:, negItemIdx]
#
#             delta  = -ui' * (posItemVec - negItemVec)
#             curHeight += log((1 + exp(delta)))
#         end
#         push!(res, curHeight)
#     end
#     return res
# end



function sigma(x)
    @assert (!isnan(x)) "input to sigma is nan"
    return 1 / (1 + exp(x))
end

# used in sort()
function compareItems(item1, item2)
    if parse(Int, split(item1, ":")[1]) > parse(Int, split(item2, ":")[1])
        return true
    else
        return false
    end
end


function plotFigure(plotDir, curTime, dataset,useCofi, algo, metric, ni, k, plotY_eval, plotY_train, plotY_obj)
    plotX = collect(1:length(plotY_obj))
    # title("minimizing loss")
    # ylabel("value of loss")

    algoName = ""
    if algo == 1
        algoName =  "rnorm"
    elseif algo ==2
        algoName =   "pnorm"
    else
        algoName =   "inorm"
    end

    metricName = ""
    if metric == 1
        metricName = "MAP"
    else
        metricName = "NDCG"
    end

    title("$algoName maximizing $metricName@$k")
    ylabel("value of map@$k")
    xlabel("iterations")
    # plot(plotX, plotY_obj, color="red", linewidth =2.0)
    grid(b=true)
    plot(plotX, plotY_eval, color="blue", label="test MAP", linewidth =2.0)
    plot(plotX, plotY_train, color = "green", label ="train MAP", linewidth =2.0)
    legend(loc=1)

    #make file name
    if useCofi == false
        filename = plotDir *  algoName *"_rand" * "_" * dataset * "_given$(ni)_" *"$curTime.svg"
    else
        filename = plotDir *  algoName * "_" * dataset * "_given$(ni)_" *"$curTime.svg"
    end

    savefig(filename)
end


"""
END Util
"""
