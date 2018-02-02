using MATLAB

include("train.jl")
include("util.jl")
# cannot calculate U *V explicitly; calculated the non-zero entries in X only
function solve_trace_reg(objf, lambda,Y ,dims, evalf, opts)


# % Partially corrective boosting
# % Solve  Min_M  func(M) + lambda * ||M||_*
# % Inputs:
# %   objf: a function handle: [f, G] = objf(U, V),
# %     where f and G are objective value and gradient of loss L at X, resp.
# %     Both X and G are matrices
# %   evalf: a function handle that evaluates the performance of the solution
# %           at each iteration (eg computes NMAE on the test set)
# %   opts: parameters for the solver
#     X: the tuple of dimensions of original matrix
#     Y: validation set
# % Outputs:
# %   solution: the solution
# %   obj: the objective value at the solution
# %   iter: number of boosting iterations
# %   msg: termination message

	m = dims[1];
	n = dims[2];
	k = opts["init_rank"]; # % Initialize U and V with rank k= 2

	U = zeros(m, k);
	V = zeros(k, n);

	total_time = 0;    # % Total time consumed by the algorithm
	local_search_time = 0; # % Time spend on local search

	# global feval_logistic time_logistic
	# feval_logistic = 0;     time_logistic = 0;

	perf = 0;
	for i in 1 : opts["max_iter"]

	  t1 = now(); # TODO
	  k = size(U, 2);

		if opts["use_local"]
		    t2 = cputime;
		    nel = k*(m+n);
		    ub = fill(Inf, nel, 1);
		    #% local search    #TODO use MATLAB.jl
		    [UV, obj, ~, ~, msg] = lbfgsb([U(:); V(:)], -ub, ub, @obj_UV,[], [], opts.lbfgsb_in);
		    U = reshape(UV(1:k*m), [m, k]);
		    V = reshape(UV(1+k*m:end), [k, n]);
		    t2 = now() - t2;
		    local_search_time = local_search_time + t2;
		end

		# this norm is the sqrt of sum among dim of rank, ie V(r,k) gives res of dim (1,k)
		  norm_U = sqrt(sum(U.*U, 1))';
		  norm_V = sqrt(sum(V.*V, 2));
		  norm_UV = sqrt(norm_U .* norm_V);
		  idx = (norm_U .> 1e-5) & (norm_V .> 1e-5);
		  nidx = sum(idx);
		if nidx < length(idx) # if there r some entries norm_U norm_V disagree
		    U = U(:, idx);
		    V = V(idx, :);
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
  		if opts.use_local || i > 1
  			msg = msg2str(msg, 'lbfgsb')
    		fprintf('iter=$i, loss = $loss, obj=$obj, perf=$perf, k=$k, time=$total_time, ls_time=local_search_time,
    			msg=$msg\n')
  		end

		if nidx > 0
    		sk = 0.5*(norm_U'*norm_U + norm_V'*norm_V); #% Xk = U*V;     #TODO ???
  		else
    		sk = 0; #% Xk = zeros(m, n); sk ???
  		end

  		if i > 1 && abs(pre_obj-obj) / minimum(abs([pre_obj, obj])) < opts["rtol"]
		    msg = 'Stop with small relative change';
		    break;
		elseif total_time > opts["max_time"]
		    msg = 'Stop by max_time';
		    break;
		elseif i == opts["max_iter"]
		    msg = 'Stop with max iteration';
		    break;
		else
		    pre_obj = obj;
		end

		# rewrite pass in the vals match nonzero entries of X()
		[~, G] = objf(U,V,Y,opts["relThreshold"])    #% G is a sparse matrix
		[u, ~, v] = svd(G, 0); #% this is polar  # matlab call
		v = -v;

		#% line search #TODO
  		[weights, obj, ~, ~, msg] = lbfgsb([1; 0.5], [0; 0], [inf; inf], ...
                                      @obj_ls, [], [], opts.lbfgsb_in); #lbfgsb_in TODO ???
  		loss = obj - lambda*(sk*weights[1] + weights[2));
		  weights = sqrt(weights);
		  if nidx > 0
		      U = [bsxfun(@times, U, weights[1]*(norm_UV./norm_U)'), weights[2] * u];
		      V = [bsxfun(@times, V, weights[1]*norm_UV./norm_V); weights[2] * v'];
		  else
		      U = weights[2] * u;
		      V = weights[2] * v';
		  end
		  t1 = cputime - t1;  total_time = total_time + t1;

		  if !isempty(evalf) && !isnan.(evalf)
		    perf = evalf(U, V, Y); #evaluate(U, V, Y; k=5, relThreshold =4, metric=1)
		  end

	end
	#TODO
	solution = U*V;
	iter = i;

	if opts.verbose
	  println('$msg\n');
	end

		#% local search objective
	  function obj_UV(M)
	    r = maximum(size(M)) / (m+n);
	    UU = reshape(M[1:r*m], [m, r]);
	    VV = reshape(M[1+r*m:end], [r, n]);

	    f, G =  objf(UU,VV,Y,opts["relThreshold"]) #TODO what is the stuff passed in
	    f = f + 0.5 * lambda * norm(M)^2; # the largest singular value of A, max(svd(A)).
	    g = hcat(vec(G*VV' + lambda*UU),
	        vec(UU'*G +lambda*VV));
	    return f, g
	  end

	  #% Line search objective
	  function obj_ls(x)
	  	tempU = hcat(sqrt(x(1))*U, sqrt(x(2))*u)
	  	tempV = vcat(sqrt(x(1))*V; sqrt(x(2))*v')
	    f, G = objf( tempU, tempV, Y, opts["relThreshold"])

	    f = f + lambda*(sk*x[1] + x[2]);
	    g = [sum(sum(U .* (G*V'),1),1)+lambda*sk
	         u'*(G*v)+lambda];
	    return f, g
	  end

	  function vec(in_v)
	    out_v = in_v[:];
	    return out_v
	  end

	return solution, obj, iter, msg
end