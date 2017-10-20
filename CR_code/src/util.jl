

"""
Util
"""
function read_from_file(path)
    res = []
    open(path) do f
        for line in eachline(f)
            push!(res, split(strip(line),' '))
        end
    end
    return res
end

function write_to_file(path, matrix)
    open(path, 'w') do f
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
function get_pos_users(X, itemId)
    res = []
    for userId in  1:size(X)[1]
        row = X[userId,:]
        for item in row
            temp = split(item, ":")
            if itemId == parse(Int,temp[1]) && temp[2] =="1"
                push!(res,userId)
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
function get_neg_users(X, itemId)
    res = []
    for userId in  1:size(X)[1]
        row = X[userId,:]
        for item in row
            temp = split(item, ":")
            if itemId == parse(Int,temp[1]) && temp[2] =="-1"
                push!(res,userId)
                break
            end
        end
    end
    return res
end

# get relevent items for user i
# @param userVec: vector of strings in forms of "itemId:rating"

function get_pos_items(userVec)
    res = []
    for item in userVec
        if split(item,":")[2] == "1"
            push!(res, item)
        end
    end
    return res
end


# get non-relevent items for user i
function get_neg_items(userVec)
    res = []
    for item in userVec
        if split(item,":")[2] == "-1"
            push!(res, item)
        end
    end
    return res
end


# @param following convention from paper, xj is V[:,j]
# TODO test, why would it return a Inf
function get_height(xj, ui, V, posItems)
    curHeight = 0
    for posItem in posItems
        posItemIdx = parse(Int,split(posItem,":")[1])
        posItemVec = V[:, posItemIdx]

        delta  = (ui' * (posItemVec - xj))[1] ## So the method cannot work with randomized U, V ???
        ri = log(2, (1 + exp(-delta)))
        #TEST
        if ri == Inf
            temp = posItemVec-xj
            temp2 = ui' * temp
            println(temp)
            println(temp2)
            println(delta)
            println(log(2, (1 + exp(-delta))))
            ri = 10000
        end
        #END TEST
        curHeight += ri

    end
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

#
#return the vector of reverse heights of relevent item for user i
function get_reverse_heights(userVec, ui, V)
    res = []
    posItems = get_pos_items(userVec)
    negItems = get_neg_items(userVec)

    for posItem in posItems
        posItemIdx = parse(Int,split(posItem, ":")[1])
        posItemVec = V[: , posItemIdx]
        curHeight = 0.0
        for negItem in negItems
            negItemIdx = parse(Int,split(negItem,":")[1])
            negItemVec = V[:, negItemIdx]

            delta  = -ui' * (posItemVec - negItemVec)
            curHeight += log(2, (1 + exp(delta)))
        end
        push!(res, curHeight)
    end
    return res
end



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
