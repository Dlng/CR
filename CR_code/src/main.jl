# using Train

include("train.jl")
using ConfParser
using PyPlot
# 1. read in config
# setup logger and trainer
# train
# eval
# output U, M, loggings

#! dimW should be consistent with ~/Codes/cofirank/config/real.cfg:dimW
function main()
    println(111)
    # configPath = ARGS[1]
    # configPath = "/home/weilong/CR/CR_code/config/default.ini"
    # configPath = "/Users/Weilong/Codes/CR/CR_code/config/default.ini"
    configPath = "/Users/Weilong/Codes/CR/CR_code/config/test.ini"
    conf = ConfParser.ConfParse(configPath)
    ConfParser.parse_conf!(conf)

    # read config
    dataset=parse(Int,retrieve(conf, "cr-train", "dataset"))

    if dataset == 1
        dataset = "yahoo"
    elseif dataset == 2
        dataset = "ml100k"
    elseif dataset == 3
        dataset = "ml1m"
    else
        dataset = "temp"
    end

    ni = parse(Int,ConfParser.retrieve(conf, dataset, "ni"))
    dimW = parse(Int,ConfParser.retrieve(conf, dataset, "dimW"))
    #algo 1= inf push, 2 = p norm, 3 = r norm
    algo = parse(Int,ConfParser.retrieve(conf, dataset, "algo"))
    p = parse(Int,retrieve(conf, dataset, "p"))
    useCofi = parse(Bool,retrieve(conf, dataset, "useCofi"))
    regval=parse(Float64,retrieve(conf, dataset, "regval"))
    learningRate=parse(Float64,retrieve(conf, dataset, "learningRate"))
    relThreshold=parse(Int,retrieve(conf, dataset, "relThreshold"))
    convThreshold=parse(Float64,retrieve(conf, dataset, "convThreshold"))
    iterNum=parse(Int,retrieve(conf, dataset, "iterNum"))
    k=parse(Int,retrieve(conf, dataset, "k"))
    metric=parse(Int,retrieve(conf, dataset, "metric"))

    # for INORM
    infGamma=parse(Int,retrieve(conf, dataset, "infGamma"))
    innerRegVal =parse(Float64,retrieve(conf, dataset, "innerRegVal"))
    epochs=parse(Int,retrieve(conf, dataset, "epochs"))
    innerLngRate=parse(Float64,retrieve(conf, dataset, "innerLngRate"))
    innerConvThreshold=parse(Float64,retrieve(conf, dataset, "convThreshold"))
    TRAIN_PATH = retrieve(conf, dataset, "TRAIN_PATH")
    VALIDATE_PATH = retrieve(conf, dataset, "VALIDATE_PATH")
    TEST_PATH = retrieve(conf, dataset, "TEST_PATH")
    U_PATH = retrieve(conf, dataset, "U_PATH")
    V_PATH = retrieve(conf, dataset, "V_PATH")
    U_OPT_PATH = retrieve(conf, dataset, "U_OPT_PATH")
    V_OPT_PATH = retrieve(conf, dataset, "V_OPT_PATH")
    figure_Dir = retrieve(conf, dataset, "figure_Dir")




    #load data set
    X = readdlm(TRAIN_PATH)
    Y = readdlm(VALIDATE_PATH)
    T = readdlm(TEST_PATH)
    # @assert size(X)[1] == m "Wrong dim in X $(size(U))"
    # @assert size(Y) == (m,10) "Wrong dim in Y $(size(V))"
    # @assert size(T)[1] == m "Wrong dim in Z $(size(U))"

    m = countlines(U_PATH)
    n = countlines(V_PATH)

    if useCofi == false
        srand(1234) # testset seed
        U = randn((dimW, m))
        V = randn((dimW, n))
    else
        U = []
        V = [] # TODO optimize
        U = read_numeric_matrix_from_file(U_PATH, dimW, m)
        V = read_numeric_matrix_from_file(V_PATH, dimW, n)

        U = U'
        V = V'
        println(size(U)[2])
        println(size(V)[2])
        # @assert size(U) == (dimW,m) "Wrong dim in U_PATH $(size(U))"
        # @assert size(V) == (dimW,n) "Wrong dim in V_PATH $(size(V))"
    end


    # print configs
    println("START PRINTING CONFIGs")
    println("dataset $dataset")
    println("ni $ni")
    println("dimW $dimW")
    println("algo $algo")
    println("p $p")
    println("useCofi $useCofi")
    println("infGamma $infGamma")
    println("regval $regval")
    println("learningRate $learningRate")
    println("relThreshold $relThreshold")
    println("convThreshold $convThreshold")
    println("iterNum $iterNum")
    println("k $k")
    println("metric $metric")
    println("epochs $epochs")
    println("innerLngRate $innerLngRate")
    println("convThreshold $convThreshold")
    println("TRAIN_PATH $TRAIN_PATH")
    println("VALIDATE_PATH $VALIDATE_PATH")
    println("TEST_PATH $TEST_PATH")
    println("U_PATH $U_PATH")
    println("V_PATH $V_PATH")
    println("U_OPT_PATH $U_OPT_PATH")
    println("V_OPT_PATH $V_OPT_PATH")
    println("figure_Dir $figure_Dir")
    println("END PRINTING CONFIGs")
    U, V, curTime = train(X ,U, V, Y, T, figure_Dir, dataset,ni, algo=algo, p=p, infGamma=infGamma  ,
    regval = regval, convThreshold=convThreshold, learningRate =learningRate, relThreshold = relThreshold,
    iterNum = iterNum, k = k, metric=metric, epochs=epochs, innerLngRate=innerLngRate, convThreshold=convThreshold )

    println("curTime $curTime")

    # write optimized U, V to file
    U_OPT_PATH = string(U_OPT_PATH,"_",curTime,".lsvm")
    V_OPT_PATH = string(V_OPT_PATH,"_",curTime,".lsvm")
    writedlm(U_OPT_PATH, U, ", ")
    writedlm(V_OPT_PATH, V, ", ")
end


main()
