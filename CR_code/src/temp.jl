include("util.jl")
include("train.jl")

function temp()

    TRAIN_PATH="/Users/Weilong/Codes/CR_data/ml-100k/train_given10.lsvm"
    VALIDATE_PATH="/Users/Weilong/Codes/CR_data/ml-100k/validate_given10.lsvm"
    TEST_PATH="/Users/Weilong/Codes/CR_data/ml-100k/test_given10.lsvm"
    U_PATH="/Users/Weilong/Codes/CR_data/ml-100k/U_given10.lsvm"
    V_PATH="/Users/Weilong/Codes/CR_data/ml-100k/V_given10.lsvm"
    relThreshold = 4
    useCofi = false
    dimW = 10
    X = readdlm(TRAIN_PATH)
    T = readdlm(TEST_PATH)
    Y = readdlm(VALIDATE_PATH)
    m = countlines(U_PATH)
    n = countlines(V_PATH)

    if useCofi == false
        srand(1234) # testset seed
        U = randn((dimW, m))
        V = randn((dimW, n))
    else
        U = []
        V = [] # TODO optimize
        U = read_numeric_matrix_from_file(U_PATH, m,dimW)
        V = read_numeric_matrix_from_file(V_PATH, n,dimW)
        U = transpose(U)
        V = transpose(V)
        debug(size(U)[2])
        debug(size(V)[2])
        @assert any(U .< 100) "U contains entry >= 1e5"
        @assert any(V .< 100) "V contains entry >= 1e5"
    end
    X, U,Y,T= preprocessing(X, U, Y, T, relThreshold)

    scoreT = evaluate(U, V, T, k=10, relThreshold=4, metric=1)
    scoreY = evaluate(U, V, Y, k=10, relThreshold=4, metric=1)
    scoreX = evaluate(U, V, X, k=10, relThreshold=4, metric=1)

    println(scoreT)
    println(scoreY)
    println(scoreX)
end


temp()
