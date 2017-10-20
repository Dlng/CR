include("../src/metric.jl")

# println(actual)
# println(typeof(actual))
# println(round(actual,2))
# println(expected_k[k])
# println(typeof(expected_k[k]))

# linear: normal
function test_dcg_k()
  y_predict = [3, 2, 3, 0 ,0, 1, 2,2, 3,0]
  expected_k = [3, 5, 6.89, 6.89, 6.89, 7.28, 7.99, 8.66, 9.61, 9.61]
  k = 10
  # actual = metric.dcg_k(y_predict ,scores,k, true)
  actual = dcg_k(y_predict ,k, true)
  @assert round(actual,2) == expected_k[k]
  println("test passed")
end

# linear: all -1
function test_dcg_k2()
  y_predict = [-1, -1 , -1 , -1, -1, -1, -1, -1, -1, -1]
  expected_k = 0.0
  k = 10
  # actual = metric.dcg_k(y_predict ,scores,k, true)
  actual = dcg_k(y_predict ,k, true)
  @assert actual == expected_k
  println("test passed")
end

# exp: normal
function test_dcg_k_e1()
  y_predict = [3, 2, 3, 0 ,0, 1, 2, 2, 3,0]
  expected_k = [7.0,8.89279,12.3928 ,12.3928 ,12.3928 ,12.749  ,13.749  ,14.6954 ,16.8026 ,16.8026]
  k = 10
  actual = dcg_k(y_predict ,k=k)
  @assert round(actual,2) == round(expected_k[k],2)
  println("test passed")
end

# exp: all -1
function test_dcg_k_e2()
  y_predict = [-1, -1 , -1 , -1, -1, -1, -1, -1, -1, -1]
  expected_k = 0.0
  k = 10
  # actual = metric.dcg_k(y_predict ,scores,k, true)
  actual = dcg_k(y_predict ,k=k)
  @assert actual == expected_k
  println("test passed")
end

# testDcg_k()

function test_avg_precision_k()
    y_predict = [3, 2, 3, 0 ,0, 1, 2,2, 3,0]
    resIdx = [1,3,9]
    k = 10
    relThreshold = 3
    res = [precision_k(y_predict, relThreshold,id) for id in resIdx]
    expected = sum(res)/size(res)[1]
    actual = avg_precision_k(y_predict, relThreshold, k)

    @assert round(actual,2) == round(expected,2)
    println("test passed")
end

function test_avg_precision_k2()
  y_predict = [-1, -1 , -1 , -1, 1, -1, -1, -1, -1, -1]
  relThreshold = 0
  k = 5
  actual = avg_precision_k(y_predict, 0, 5)
  expected = 0.2
  @assert round(actual,2) == round(expected,2)
  println("test passed")
end

test_avg_precision_k()
