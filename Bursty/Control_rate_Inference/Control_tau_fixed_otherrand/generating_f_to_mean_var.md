$G(z)=\Sigma z^mP(m)$

$G'(z)=\Sigma mz^{m-1}P(m)$

$G''(z)=\Sigma m(m-1)z^{m-2}P(m)$

$E(x)=\Sigma xP(x)=G'(1)$

$D(x)=E(x^2)-E(x)^2=x^2P(x)-[xP(x)]^2$

$G''(1)=\Sigma m^2P(m)-\Sigma mP(m)=\Sigma m^2P(m)-G'(1)$

$D(x)=G''(1)+G'(1)-[G'(1)]^2$


e.g for bursty model

$G(z)=\text{exp}(\frac{a\tau b(z-1)}{1-b(z-1)})$

$E(x)=a\tau b$

$D(x)=a\tau b+2a\tau b^2$
