using MATLAB

# cannot calculate U *V explicitly; calculated the non-zero entries in X only
function solve_trace_reg(X,Y,T,eval_obj,eval_gradient, evalf, opts)


# % Partially corrective boosting
# % Solve  Min_M  func(M) + lambda * ||M||_*
# % Inputs:
# %   objf: a function handle: [f, G] = objf(U, V),
# 		eval_obj
#		eval_gradient
# %     where f and G are objective value and gradient of loss L at X, resp.
# %     Both X and G are matrices
# %   evalf: a function handle that evaluates the performance of the solution
# %           at each iteration (eg computes NMAE on the test set)
# %   opts: parameters for the solver
#     X: the tuple of dimensions of original matrix
#     Y: validation set
# % Outputs:
# %   U, V : the solution
# %   obj: the objective value at the solution
# %   iter: number of boosting iterations
# %   msg: termination message
	lambda = opts["lambda"]
	dims = opts["dims"]
	m = dims[1];
	n = dims[2];
	k = opts["init_rank"]; # % Initialize U and V with rank k= 2

	U = zeros(m, k);
	V = zeros(k, n);

	total_time = 0;    # % Total time consumed by the algorithm
	local_search_time = 0; # % Time spend on local search

	# global feval_logistic time_logistic
	# feval_logistic = 0;     time_logistic = 0;

	curEvalVali = 0
    preEvalVali = 0

	# plotting

    plotY_obj = [] # eval obj on training set using updated U V
    plotY_train = [] # eval metric on training set using updated U V
    plotY_eval = [] # eval metric on testing set using updated U V
	for i in 1 : opts["max_iter"]
	  debug("In GCG, in iter $i")
	  t1 = Dates.value(now()); 
	  k = size(U, 2);

		if opts["use_local"]
		    t2 = Dates.value(now());
		    nel = k*(m+n);
		    ub = fill(Inf, nel, 1);
		    #% local search    

		    UV, obj, _, _, msg = mxcall(:lbfgsb_mex, 5, vcat(U[:], V[:]), -ub, ub, obj_UV,[], [], opts["lbfgsb_in"]);
		    # (UV, obj, ~, ~, msg) = mxcall(:lbfgsb_mex, 5, vcat(U(:), V(:)), -ub, ub, obj_UV);
		    U = reshape(UV[1:k*m], [m, k]);
		    V = reshape(UV[1+k*m:end], [k, n]);
		    t2 = Dates.value(now()) - t2;
		    local_search_time = local_search_time + t2;
		end

		# this norm is the sqrt of sum among dim of rank, ie V(r,k) gives res of dim (1,k)
		  norm_U = sqrt(sum(U.*U, 1))';
		  norm_V = sqrt(sum(V.*V, 2));
		  norm_UV = sqrt(norm_U .* norm_V);
		  idx = (norm_U .> 1e-5) & (norm_V .> 1e-5);
		  nidx = sum(idx);
		if nidx < length(idx) # if there r some entries norm_U norm_V disagree
		    U = U[:, idx];
		    V = V[idx, :];
		    norm_U = norm_U(idx);
		    norm_V = norm_V(idx);
		    norm_UV = norm_UV(idx);
		end
  		if opts["use_local"]
		    loss = obj - 0.5*lambda*(norm_U'*norm_U + norm_V'*norm_V);
		    if isempty(loss)
		      loss = obj;
		    end
  		end
  		if opts["use_local"] || i > 1
  			# msg = msg2str(msg, 'lbfgsb')
    	# 	println('iter=$i, loss = $loss, obj=$obj, curEvalVali=$curEvalVali, 
    	# 		curEvalTest=$curEvalTest, curEvalTrain=$curEvalTrain, k=$k, time=$total_time, ls_time=local_search_time,
    	# 		msg=$msg\n')
  		end

		if nidx > 0
    		sk = 0.5*(norm_U'*norm_U + norm_V'*norm_V); #% Xk = U*V;    
  		else
    		sk = 0; #% Xk = zeros(m, n); sk ???
  		end

  		if i > 1 && abs(pre_obj-obj) / minimum(abs([pre_obj, obj])) < opts["rtol"]
		    msg = "Stop with small relative change";
		    break;
		elseif total_time > opts["max_time"]
		    msg = "Stop by max_time";
		    break;
		elseif i == opts["max_iter"]
		    msg = "Stop with max iteration";
		    break;
		else
		    pre_obj = obj;
		end

		# rewrite pass in the vals match nonzero entries of X()
		G = eval_gradient(U,V,Y,opts) #    #% G is a sparse matrix
		u, _, v = mxcall(:svd,3,G,0);  #% this is polar  
		v = -v;

		#% line search 
  		weights, obj, _, _, msg = mxcall(:lbfgsb_mex, 5,vcat(1, 0.5), vcat(0, 0), vcat(Inf, Inf),obj_ls, [], [], opts["lbfgsb_in"]); 
    	# weights, obj, _, _, msg = mxcall(:lbfgsb_mex,5, vcat(1, 0.5), vcat(0, 0), vcat(Inf, Inf),obj_ls);
  		loss = obj - lambda*(sk*weights[1] + weights[2]);
		  weights = sqrt(weights);
		  if nidx > 0 
		  	preTempU = weights[1]*(norm_UV./norm_U)'
		  	preTempV = weights[1]*norm_UV./norm_V
		  	tempU = mxcall(:bsxfun, 1,mat"""@times""", U, preTempU)
			tempV = mxcall(:bsxfun, 1,mat"""@times""", V, preTempV)		  	
		  	U = hcat(tempU, weights[2] * u)
		  	V = vcat(tempV, weights[2] * v')
		  else
		      U = weights[2] * u;
		      V = weights[2] * v';
		  end
		  t1 = Dates.value(now()) - t1; 
		  total_time = total_time + t1;

		  if !isempty(evalf) && !isnan.(evalf)
		    curEvalVali = evaluate(U, V, Y) # using MAP@5
        	curEvalTest = evaluate(U, V, T, 
        		k = opts["k"], relThreshold=opts["relThreshold"],metric =opts["metric"] )
        	curEvalTrain = evaluate(U, V, X, 
        		k = opts["k"], relThreshold=opts["relThreshold"],metric =opts["metric"])
        	curVal_obj = eval_obj(U, V, X,opts)
        	push!(plotY_obj,curVal_obj)
        	push!(plotY_train,curEvalTrain)
        	push!(plotY_eval, curEvalTest)
		  end

	end

	iter = i;

	# if opts["verbose"]
	  # println('$msg\n');
	# end

		#% local search objective
	  function obj_UV(M)
	    r = maximum(size(M)) / (m+n);
	    UU = reshape(M[1:r*m], [m, r]);
	    VV = reshape(M[1+r*m:end], [r, n]);
	    f = eval_obj(UU,VV,Y,opts)
	    G = eval_gradient(UU,VV,Y,opts)
	    # f, G =  objf(UU,VV,Y,opts["relThreshold"]) #TODO what is the stuff passed in
	    f = f + 0.5 * lambda * norm(M)^2; # the largest singular value of A, max(svd(A)).
	    g = hcat(vec(G*VV' + lambda*UU),
	        vec(UU'*G +lambda*VV));
	    return f, g  
	  end

	  #% Line search objective
	  function obj_ls(x)
	  	tempU = hcat(sqrt(x[1])*U, sqrt(x[2])*u)
	  	tempV = vcat(sqrt(x[1])*V; sqrt(x[2])*v')
	  	f = eval_obj( tempU, tempV, Y,opts)
	    G = eval_gradient( tempU, tempV, Y,opts)
	    # f, G = objf( tempU, tempV, Y, opts["relThreshold"])

	    f = f + lambda*(sk*x[1] + x[2]);
	    g = [sum(sum(U .* (G*V'),1),1)+lambda*sk
	         u'*(G*v)+lambda];
	    return f, g
	  end

	  function vec(in_v)
	    out_v = in_v[:];
	    return out_v
	  end

	return U, V, plotY_obj,plotY_train,plotY_eval
end
