using OptimalTransport,TaylorSeries,Distributions,Distances

function bursty(N,a,b,τ)
    f(u) = exp(a*b*τ*u/(1-b*u));
    taylorexpand = taylor_expand(x->f(x),-1,order=N);
    P = zeros(N)
    for j in 1:N
        P[j] = taylorexpand[j-1]
    end
    return P
end;

temp1 = bursty(60,0.0282,3.46,120)
temp2 = bursty(60,0.0182,2.46,120)
μ = Categorical(vcat(temp1,1-sum(temp1)))
ν = Categorical(vcat(temp2,1-sum(temp2)))
cost = ot_cost(sqeuclidean, μ, ν)