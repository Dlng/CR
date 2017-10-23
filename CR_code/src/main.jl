# using Train

include("./train.jl")
using ConfParser

# 1. read in config
# setup logger and trainer
# train
# eval
# output U, M, loggings

#! dimW should be consistent with ~/Codes/cofirank/config/real.cfg:dimW
function main()
    println(111)
    # configPath = ARGS[1]
    # configPath = "config/default.ini"
    configPath = "/Users/Weilong/Documents/UWaterloo/4A/URA/CR_code/config/default.ini"
    conf = ConfParser.ConfParse(configPath)
    ConfParser.parse_conf!(conf)

    # read config
    u = parse(Int, ConfParser.retrieve(conf, "cr-train", "u"))
    m = parse(Int,ConfParser.retrieve(conf, "cr-train", "m"))
    dimW = parse(Int,ConfParser.retrieve(conf, "cr-train", "dimW"))
    #algo 1= inf push, 2 = revese height, 3 = p norm 4= new
    algo = parse(Int,ConfParser.retrieve(conf, "cr-train", "algo"))
    p = parse(Int,retrieve(conf, "cr-train", "p"))
    useCofi = parse(Bool,retrieve(conf, "cr-train", "useCofi"))
    infGamma=parse(Int,retrieve(conf, "cr-train", "infGamma"))
    regval=parse(Float64,retrieve(conf, "cr-train", "regval"))
    learningRate=parse(Float64,retrieve(conf, "cr-train", "learningRate"))
    relThreshold=parse(Int,retrieve(conf, "cr-train", "relThreshold"))
    TRAIN_PATH = retrieve(conf, "cr-train", "TRAIN_PATH")
    VALIDATE_PATH = retrieve(conf, "cr-train", "VALIDATE_PATH")
    TEST_PATH = retrieve(conf, "cr-train", "TEST_PATH")
    U_PATH = retrieve(conf, "cr-train", "U_PATH")
    V_PATH = retrieve(conf, "cr-train", "V_PATH")
    U_OPT_PATH = retrieve(conf, "cr-train", "U_OPT_PATH")
    V_OPT_PATH = retrieve(conf, "cr-train", "V_OPT_PATH")
    # TODO work with randomly initialized U, V, later with those from cofirank
    if useCofi == false
        srand(1234) # testset seed
        U = randn((dimW, u))
        V = randn((dimW, m))
    else
        U = []
        V = []
        U = readdlm(U_PATH)
        V = readdlm(V_PATH)

        U = U'
        V = V'
        @assert size(U) == (dimW,u) "Wrong dim in U_PATH $(size(U))"
        @assert size(V) == (dimW,m) "Wrong dim in V_PATH $(size(V))"
    end

    #load data set
    X = readdlm(TRAIN_PATH)
    Y = readdlm(VALIDATE_PATH)
    T = readdlm(TEST_PATH)

    U, V = train(X ,U, V, Y, T, algo=algo, p=p, infGamma=infGamma  ,
    regval = regval, dimW = dimW, learningRate =learningRate, relThreshold = relThreshold)

    # write optimized U, V to file
    writedlm(U_OPT_PATH, U, ", ")
    writedlm(V_OPT_PATH, V, ", ")
end


main()
