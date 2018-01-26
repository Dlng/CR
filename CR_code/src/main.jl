include("train.jl")
using ConfParser
using Logging
using PyPlot
# using PrintLog
# 1. read in config
# setup logger and trainer
# train
# eval
# output U, M, loggings

#! dimW should be consistent with ~/Codes/cofirank/config/real.cfg:dimW
function main()
    debug(111)
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
    log_path = retrieve(conf, dataset, "log_path")

    curTime = Dates.value(now())
    #set up logger
    log_file = log_path * dataset * "_$algo"  * "_$useCofi"  * "_given$ni"  * "_$curTime" *".txt"

    Logging.configure(level=DEBUG, filename=log_file)
    # Logging.configure(level=OFF) # turn logger off



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
        U = read_numeric_matrix_from_file(U_PATH, m,dimW)
        V = read_numeric_matrix_from_file(V_PATH, n,dimW)
        U = transpose(U)
        V = transpose(V)
        debug(size(U)[2])
        debug(size(V)[2])
        @assert any(U .< 100) "U contains entry >= 1e5"
        @assert any(V .< 100) "V contains entry >= 1e5"
    end


    # print configs
    debug("START PRINTING CONFIGs")
    debug("dataset $dataset")
    debug("ni $ni")
    debug("dimW $dimW")
    debug("algo $algo")
    debug("p $p")
    debug("useCofi $useCofi")
    debug("infGamma $infGamma")
    debug("regval $regval")
    debug("learningRate $learningRate")
    debug("relThreshold $relThreshold")
    debug("convThreshold $convThreshold")
    debug("iterNum $iterNum")
    debug("k $k")
    debug("metric $metric")
    debug("epochs $epochs")
    debug("innerLngRate $innerLngRate")
    debug("innerConvThreshold $innerConvThreshold")
    debug("TRAIN_PATH $TRAIN_PATH")
    debug("VALIDATE_PATH $VALIDATE_PATH")
    debug("TEST_PATH $TEST_PATH")
    debug("U_PATH $U_PATH")
    debug("V_PATH $V_PATH")
    debug("U_OPT_PATH $U_OPT_PATH")
    debug("V_OPT_PATH $V_OPT_PATH")
    debug("figure_Dir $figure_Dir")
    debug("curTime $curTime")
    debug("END PRINTING CONFIGs")
    U, V= train(X ,U, V, Y, T, figure_Dir, dataset,useCofi, curTime, ni, algo=algo, p=p, infGamma=infGamma  ,
    regval = regval, convThreshold=convThreshold, learningRate =learningRate, relThreshold = relThreshold,
    iterNum = iterNum, k = k, metric=metric, epochs=epochs, innerLngRate=innerLngRate, innerConvThreshold=innerConvThreshold )

    # write optimized U, V to file
    if useCofi == false
        U_OPT_PATH = string(U_OPT_PATH,"_","rand")
        V_OPT_PATH = string(V_OPT_PATH,"_","rand")
    end
    U_OPT_PATH = string(U_OPT_PATH,"_",curTime,".lsvm")
    V_OPT_PATH = string(V_OPT_PATH,"_",curTime,".lsvm")
    writedlm(U_OPT_PATH, U, ", ")
    writedlm(V_OPT_PATH, V, ", ")
end


main()
