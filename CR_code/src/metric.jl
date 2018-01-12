
# ranking metrics

# dcg with exponential discount
function dcg_k( ranking; k=10)
    sum =0
    for idx in 1:k
      if idx <= length(ranking)
        sum += (2^(max(0,ranking[idx])) -1) / log2(1 + idx)
      end
    end
    return sum
end

# dcg with linear discount
function dcg_k( ranking, k, isLinear::Bool)
  sum = max(0,ranking[1])
  for idx in 2:k
    if idx <= length(ranking)
      sum += max(0,ranking[idx])/log(idx)
    end
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
    if r <= length(y_predict)
      if y_predict[r] >= relThreshold
        push!(relIdxs, r)
      end
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
