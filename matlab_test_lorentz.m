a=3;
x=4;
x0=2;
gamma=2.2;

v=1/(1+(2*(x-x0)/gamma).^2);
v2=v*v;
gamma2=gamma*gamma;
av=8*a*v*v;

j1=v;
j2=-av*(x-x0)/gamma2;
j3=av*(x-x0)*(x-x0)/(gamma2*gamma);


delta=1e-10;
jj1=(lorentz(x,x0,a+delta,gamma)-lorentz(x,x0,a,gamma))/delta;
jj2=(lorentz(x,x0+delta,a,gamma)-lorentz(x,x0,a,gamma))/delta;
jj3=(lorentz(x,x0,a,gamma+delta)-lorentz(x,x0,a,gamma))/delta;





function y=lorentz(x,x0,a,gamma)

    y=a/(1+(2*(x-x0)/gamma).^2);

end