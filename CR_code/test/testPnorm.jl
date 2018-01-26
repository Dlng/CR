include("../src/util.jl")
include("../src/p-norm.jl")


function test_get_p_norm_gradient_by_user()
    X = readdlm("/Users/Weilong/Desktop/Webscope_R1/temp/train.lsvm")
    srand(1234) # test: set seed
    u = 10
    m = 11
    dimW = 10
    U = randn((dimW, u))
    V = randn((dimW, m))
    userNum = u
    userVec = ["1:-1" ,"2:-1" ,"3:-1" ,"4:1" ,"6:-1" ,"8:-1" ,"9:1" ,"11:-1"]
    ui = U[:, 1]
    res = get_p_norm_gradient_by_user(userVec, ui, U, V,2)
    debug(res)
    debug(typeof(res))
    assert(size(res)[1] != 0 )
end

function test_get_p_norm_gradient_by_item()
    X = readdlm("/Users/Weilong/Desktop/Webscope_R1/temp/train.lsvm")
    itemId = 2
    srand(1234) # test: set seed
    u = 50
    m = 50
    dimW = 10
    U = randn((dimW, u))
    V = randn((dimW, m))
    vh = V[:, itemId]
    userNum = u
    res = get_p_norm_gradient_by_item(X, U, V, itemId,2)
    debug(res)
    debug(vh)
    debug(typeof(res))

    assert(size(res)[1] != 0 )
    assert(vh != res)
    debug("test passed")
end


test_get_p_norm_gradient_by_item()
