include("../src/util.jl")

function test_get_pos_items()
    userVec = ["1:-1" ,"2:-1" ,"3:-1" ,"4:1" ,"6:-1" ,"8:-1" ,"9:1" ,"11:-1"]
    actual =  get_pos_items(userVec)
    expected = ["4:1" ,"9:1"]
    assert(actual == expected)
    debug("test passed")
end

function test_get_neg_items()
    userVec = ["1:-1" ,"2:-1" ,"3:-1" ,"4:1" ,"6:-1" ,"8:-1" ,"9:1" ,"11:-1"]
    actual =  get_neg_items(userVec)
    expected = ["1:-1" ,"2:-1" ,"3:-1","6:-1" ,"8:-1" ,"11:-1"]
    assert(actual == expected)
    debug("test passed")
end


function test_get_height()
    dimW = 10
    u = 49
    m = 50
    srand(1234) # test: set seed
    U = randn((dimW, u))
    V = randn((dimW, m))

    ui = U[:, 1]
    xj = V[:, 1]
    posItems = ["4:1" ,"9:1"]
    debug(get_height(xj,ui, V, posItems))
end

function test_get_pos_users()
    X = readdlm("/Users/Weilong/Desktop/Webscope_R1/temp/train.lsvm")
    itemId = 2
    res = get_pos_users(X, itemId)
    debug(size(res))
end

function test_get_height()
    posItems = ["41:1"]
    negItems = ["1:-1","4:-1","6:-1","7:-1","12:-1","16:-1","19:-1","27:-1","38:-1"]
    ui = [-2.09,-1.66,-1.20,1.97,1.48,-2.15,-1.96,2.34,1.65,4.24]
end



######################################
function runTests()
    # TRAIN_PATH=/Users/Weilong/Desktop/Webscope_R1/temp/train.lsvm
    # VALIDATE_PATH=/Users/Weilong/Desktop/Webscope_R1/temp/validate.lsvm
    # TEST_PATH=/Users/Weilong/Desktop/Webscope_R1/temp/validate.lsvm
    test_get_height()
end


test_get_pos_users()
