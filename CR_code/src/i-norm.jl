include("util.jl")
include("metric.jl")


function eval_obj(U, V, X, relThreshold)
    finalRes = 0
    for id in 1:size(X)[1]
        userVec = X[id, :]
        ui = U[:, id]
        ni = length(userVec)
        @assert (ni != 0) "INORM:eval_obj: ni is 0"
        userRes = 0
        posItems = get_pos_items(userVec, relThreshold)
        negItems = get_neg_items(userVec, relThreshold)
        # find the max over j and reuse in following loop
        maxJ = 0
        for negItem in negItmes
            negItemIdx = parse(Int,split(negItem,":")[1])
            vj = V[:, negItemIdx]
            temp = dot(ui, vj)
            if temp > maxJ
                maxJ = temp
            end
        end

        for posItem in posItems
            posItemIdx = parse(Int,split(posItem,":")[1])
            vk = V[:, posItemIdx]
            temp = maxJ + dot(-ui, vk)
            if temp > 100
                userRes += temp
            else
                userRes += log(1+ exp(temp))
            end
        end
        finalRes += 1/ni * userRes
    end
    return finalRes
end




# stop We stopped either when
# the improvement in the value of the optimization problem
# was smaller than 0.01 or after 25 iterations.

# TODO
function i_norm_optimizer(X, U, V, Y, learningRate=0.01, regval=0.1, infGamma=10,
relThreshold= 4, iterNum=200, k = 5, metric=2)

end
