include("../../../utils.jl")

sigma_on = 0.003
sigma_off = 0.004
rho_on = 0.3
w1(u) = (-(sigma_on+sigma_off+rho_on*(1-u))-sqrt((sigma_on+sigma_off+rho_on*(1-u))^2-4*sigma_on*rho_on*(1-u)))/2
w2(u) = (-(sigma_on+sigma_off+rho_on*(1-u))+sqrt((sigma_on+sigma_off+rho_on*(1-u))^2-4*sigma_on*rho_on*(1-u)))/2
G(u) = (w2(u)*exp(w1(u))-w1(u)*exp(w2(u)))/(w2(u)-w1(u))+rho_on*sigma_on*(u-1)*(exp(w2(u))-exp(w1(u)))/((sigma_off+sigma_on)*(w2(u)-w1(u)))

A(u) = (rho_on*sigma_on*(u-1)/(sigma_off+sigma_on)-w2(u))/(exp(-w1(u))*(w1(u)-w2(u)))
B(u) = (w1(u)-rho_on*sigma_on*(u-1)/(sigma_off+sigma_on))/(exp(-w2(u))*(w1(u)-w2(u)))
u = 20
A(u)+B(u)
G(u)
u = 20
A(u)*exp(-w1(u))+B(u)*exp(-w2(u))
A(u)*w1(u)*exp(-w1(u))+B(u)*w2(u)*exp(-w2(u))
(u-1)*rho_on*sigma_on/(sigma_off+sigma_on)
τ

τ = 120
w1(u) = (-(sigma_on+sigma_off+rho_on*(1-u))-sqrt((sigma_on+sigma_off+rho_on*(1-u))^2-4*sigma_on*rho_on*(1-u)))/2
w2(u) = (-(sigma_on+sigma_off+rho_on*(1-u))+sqrt((sigma_on+sigma_off+rho_on*(1-u))^2-4*sigma_on*rho_on*(1-u)))/2

G(u) = (w2(u)*exp(τ*w1(u))-w1(u)*exp(τ*w2(u)))/(w2(u)-w1(u))+rho_on*sigma_on*(u-1)*(exp(τ*w2(u))-exp(τ*w1(u)))/((sigma_off+sigma_on)*(w2(u)-w1(u)))
A(u) = (rho_on*sigma_on*(u-1)/(sigma_off+sigma_on)-w2(u))/(exp(-τ*w1(u))*(w1(u)-w2(u)))
B(u) = (w1(u)-rho_on*sigma_on*(u-1)/(sigma_off+sigma_on))/(exp(-τ*w2(u))*(w1(u)-w2(u)))
u = 2
A(u)+B(u)
G(u)

G(u) = A(u)+B(u)
τ
N = 100
taylorexpand = taylor_expand(x->G(x),0,order=N)
P = zeros(N)
for j in 1:N
    P[j] = taylorexpand[j-1]
end
P
plot(P,lw=3,line=:dash)


k1 = sigma_off-sigma_on + 2*im*sqrt(sigma_off*sigma_on)
k2 = sigma_off-sigma_on - 2*im*sqrt(sigma_off*sigma_on)

M(s,i) = sum([real(sum([round(binomial(l,w)*binomial(l,i-w)*(-1)^i*factorial(big(i))/
factorial(big(2*l+s))*((rho_on/2+k1/2)^(l-w))*((rho_on/2+k2/2)^(l-i+w)),digits=2) for w=maximum([0,i-l]):minimum([l,i])])) 
for l=Int(ceil(i/2)):30])

m = 0
P0 = (exp(-(sigma_off+sigma_on+rho_on)/2)/factorial(big(m)))*(rho_on/2)^m*(((sigma_off+sigma_on+rho_on)/2-
rho_on*sigma_on/(sigma_off+sigma_on))*sum([binomial(m,i)*M(1,i) for i=0:m])+sum([binomial(m,i)*M(0,i) for i=0:m]))

P_tele(m) = (exp(-(sigma_off+sigma_on+rho_on)/2)/factorial(big(m)))*(rho_on/2)^m*(((sigma_off+sigma_on+rho_on)/2-
rho_on*sigma_on/(sigma_off+sigma_on))*sum([binomial(m,i)*M(1,i) for i=0:m])+sum([binomial(m,i)*M(0,i) for i=0:m])
+m*(sigma_on-sigma_off)/(sigma_off+sigma_on)*sum([binomial(m-1,i)*M(1,i) for i=0:m-1]))

P1_end = [P_tele(m) for m=1:40]
plot([P0;P1_end])

