loghyper0=[0;0;0;0;log(0.1)];
covfunc={'covSum',{'covSEard','covNoise'}};

loghyper=minimize(loghyper0,'gpr',-100,covfunc,X,y)

[mu,S2]=gpr(loghyper,covfunc,X,y,X);

plot([1:5],mu,'LineWidth',3)
% hold on
% S2 = S2 - exp(2*loghyper(3));
% errorbar([1:5], mu, 2*sqrt(S2), 'g');
% plot([1:5],mu,'k+', 'MarkerSize', 17)
% hold off