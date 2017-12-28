

"""
Util
"""
function read_numeric_matrix_from_file(path, dimW, m)
    res = zeros(m,dimW)
    rowCnt = 1
    open(path, "r") do f
        for line in eachline(f)
            # println(typeof(line))
            # println(line)
            # println(isempty(line))
            # println(length(line))
            # if line == ""
            if length(line) == 1 || line == ""
                res[rowCnt, :] = randn(dimW)
            else
                tempLine = split(strip(line),' ')
                tempLine2 = [parse(Float64, x) for x in tempLine] #TODO error parse("")
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
        if parse(Int,split(item,":")[2]) >= relThreshold
            push!(res, item)
        end
    end
    return res
end


# get non-relevent items for user i
function get_neg_items(userVec, relThreshold)
    res = []
    for item in userVec
        if parse(Int,split(item,":")[2]) < relThreshold
            push!(res, item)
        end
    end
    return res
end


# @param following convention from paper, xj is V[:,j]
function get_height(xj, ui, V, posItems)
    curHeight = 0
    for posItem in posItems
        posItemIdx = parse(Int,split(posItem,":")[1])
        posItemVec = V[:, posItemIdx]
        delta = dot(ui, (posItemVec - xj))
        ri = 0
        # println("delta is $delta")
        if delta > -100 # FIX
            ri = log(1 + exp(-delta))
        else
            ri = -delta
        end

        #TEST
        if ri == Inf || isnan(ri) || isnan(curHeight)
            temp = posItemVec-xj
            temp2 = ui' * temp # why this is NAN
            println("in get height ")
            println(ri)
            println(temp)
            println(temp2)
            println(delta)

            println(size(ui) == size(temp))

            println(ui)
            println(posItemVec)
            println(xj)
            println(" ")
            # println(log((1 + exp(-delta))))
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


function get_reverse_height(xk, ui, V, negItems)
    curRHeight = 0
    for negItem in negItems
        negItemIdx = parse(Int,split(negItem,":")[1])
        negItemVec = V[:, negItemIdx]
        delta = dot(ui, (xk - negItemVec))
        ri = 0
        # println("delta is $delta")
        if delta > -100 # FIX
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



"""
END Util
"""
