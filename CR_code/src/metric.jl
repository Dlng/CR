
# ranking metrics

# dcg with exponential discount
function dcg_k( ranking; k=10)
    sum =0
    for idx in 1:k
      sum += (2^(max(0,ranking[idx])) -1) / log2(1 + idx)
    end
    return sum
end

# dcg with linear discount
function dcg_k( ranking, k, isLinear::Bool)
  sum = max(0,ranking[1])
  for idx in 2:k
      sum += max(0,ranking[idx])/log(idx)
  end
  return sum
end


# NOTE dcg does not take neg vals
# @param y_true argsorts scores in descending order
# @param y_predict the ranking of items from top to bottom
function ndcg_k(y_true, y_predict, k)
    @assert (any(isnan, y_predict) == false) "there NaN in ndcg_k"
    @assert (any(isnan, y_true) == false) "there NaN in ndcg_k"
    return dcg_k(y_predict, k = k)/dcg_k(y_true, k=k)
end

# precision at k
function precision_k(y_predict, relThreshold, k)
  count = 0
  for i in 1:k
    if y_predict[i] >= relThreshold
      count += 1
    end
  end
  return count / k
end


# @param y_predict the ranking of items from top to bottom
# @ param relThreshold: binary thresholds
# @ param k, rank thresholds
function avg_precision_k(y_predict, relThreshold, k)
  @assert (any(isnan, y_predict) == false) "there NaN in ap_k"
  relIdxs = []
  # find positions of rel items
  for r in 1:k
    if y_predict[r] >= relThreshold
      push!(relIdxs, r)
    end
  end
  if length(relIdxs) == 0
    return 0
  end
  # get precision_k for each of them
  sum = 0
  for id in relIdxs
    sum += precision_k(y_predict, relThreshold, id)
  end
  return sum/ length(relIdxs)
end



#  TEST check if the improvement in MAP@5 is smaller then threshold (i.e 0.0001)
#: FOR methods except inf_norm
# X, Y are sparse, U, V are non sparse
# size(U) = dimW x m ;
# size(V) = dimW x n
# function is_converged(X, U, V, Y, preVal; threshold=0.0001)
#     res = 0
#     for userId in 1:size(X)[1]
#       userVec = X[userId, :]
#       testUserVec = Y[userId, :]
#       # get items in Y that appears in X
#       userItemsIds = [split(item, ":")[1] for item in userVec]
#       testUserItemsIds = [split(item, ":")[1] for item in testUserVec]
#       commonIds = intersect(userItemsIds,testUserItemsIds)
#       if size(commonIds,1) == 0
#         continue
#       end
#       # get corresponding items from Y
#       commonItems = [item for item in testUserVec if split(item,":")[1] in commonIds]
#       commonItemsScores = [parse(Int,split(item, ":")[2]) for item in commonItems]
#
#       # calculate AP@5
#       ui = U[:, userId]
#       predictions = [ui' * V[:,id] for id in commonIds]
#       preds = sort!([(predVal, id) for (id, predVal) in enumerate(predictions)])
#       y_predict = [item[2] for item in preds]
#       ap = avg_precision_k(y_predict, commonItemsScores, 0, 5)
#       res += ap
#     end
#
#     map_5 = res / size(U)[2]
#     if (map_5 - preVal) >= threshold
#       return false
#     else
#       return true
#     end
#
# end
