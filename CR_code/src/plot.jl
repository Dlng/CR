using ConfParser
using PyPlot
include("util.jl")
include("train.jl")


function plot()
    """
    for one metric, on a dataset, varies the k and plot graphs for all algos
    """
    # read config
    configPath = "/Users/Weilong/Codes/CR/CR_code/config/plot.ini"
    conf = ConfParser.ConfParse(configPath)
    ConfParser.parse_conf!(conf)
    dataset=parse(Int,retrieve(conf, "plot", "dataset"))
    metric=parse(Int,retrieve(conf, "plot", "metric"))
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
    #algo 1= rnorm, 2 = p norm, 3 = i norm
    # algo = parse(Int,ConfParser.retrieve(conf, dataset, "algo"))
    algo = [2]

    TEST_PATH = retrieve(conf, dataset, "TEST_PATH")
    U_INORM_PATH = retrieve(conf, dataset, "U_INORM_PATH")
    V_INORM_PATH = retrieve(conf, dataset, "V_INORM_PATH")
    U_PNORM_PATH = retrieve(conf, dataset, "U_PNORM_PATH")
    V_PNORM_PATH = retrieve(conf, dataset, "V_PNORM_PATH")
    U_RNORM_PATH = retrieve(conf, dataset, "U_RNORM_PATH")
    V_RNORM_PATH = retrieve(conf, dataset, "V_RNORM_PATH")
    figure_Dir = retrieve(conf, dataset, "figure_Dir")
    TEST_PATH = TEST_PATH * ni * ".lsvm"

    curTime = Dates.value(now())
    out_figure_name = ""




    #load data set
    T = readdlm(TEST_PATH)

    U_i_opt = read_numeric_matrix_from_file(U_INORM_PATH, m,dimW)
    V_i_opt = read_numeric_matrix_from_file(R_INORM_PATH, n,dimW)
    U_i_opt = transpose(U_i_opt)
    V_i_opt = transpose(V_i_opt)

    U_p_opt = read_numeric_matrix_from_file(U_PNORM_PATH, m,dimW)
    V_p_opt = read_numeric_matrix_from_file(V_PNORM_PATH, n,dimW)
    U_p_opt = transpose(U_p_opt)
    V_p_opt = transpose(V_p_opt)

    U_r_opt = read_numeric_matrix_from_file(U_RNORM_PATH, m,dimW)
    V_r_opt = read_numeric_matrix_from_file(V_RNORM_PATH, n,dimW)
    U_r_opt = transpose(U_r_opt)
    V_r_opt = transpose(V_r_opt)





    # print configs
    println("START PRINTING CONFIGs")
    println("algo $algo")
    println("metric $metric")
    println("dataset $dataset")
    println("TEST_PATH $TEST_PATH")
    println("U_INORM_PATH $U_INORM_PATH")
    println("V_INORM_PATH $V_INORM_PATH")
    println("U_PNORM_PATH $U_PNORM_PATH")
    println("V_PNORM_PATH $V_PNORM_PATH")
    println("U_RNORM_PATH $U_RNORM_PATH")
    println("V_RNORM_PATH $V_RNORM_PATH")
    println("figure_Dir $figure_Dir")
    println("END PRINTING CONFIGs")

    res_pnorm = []
    res_inorm = []
    res_rnorm = []
    ks = [1, 2, 3, 4, 5, 10]
    algos = [1,2,3]
    # for each algo
    for algo in algos
    # for each k
        for k in ks
        # get metric score

        end
    end
    # plot
end
