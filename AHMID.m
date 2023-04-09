function [Z,S,E,N]=AHMID(X,D,alpha,lambda,display)

if nargin<5
    display = false;
end
epsilon=1e-6;
% maxIter =300;
maxIter =1e3;
mu = 1e-2;
mu_max = 1e12;
rou = 1.1;

DF_norm=norm(D,'fro');
b=(DF_norm)^2;
[dim,num] = size(X);
numD = size(D,2);

Z=zeros(numD,num);
% J=zeros(numD,num);
S=zeros(dim,num);
E=zeros(dim,num);
N=zeros(dim,num);
L=zeros(dim,num);
Y1=zeros(dim,num);
Y2=zeros(numD,num);
Y3=zeros(dim,num);
DtD=D'*D;
ID=eye(numD);
A = (1/num)*ones(num);
DA_Norm=norm(A,'fro');
a=DA_Norm^2;
% IE=eye(num);
iter = 0;



while iter<maxIter
    iter = iter + 1;

    %update J
    temp=Z+Y2/mu;
    temp1=svd_threshold(temp,1/mu);
    J1=temp1;
    
    %updata S
    temp=(mu*(X-D*Z-E-N+L)+Y1-Y3)/(2*alpha*(a+b)+2*mu);
    S1=temp;     
    
    %updata L
    temp=S+Y3/mu;
    L1=solve_l1l2(temp,1/mu);
    
    %update Z
    temp=(DtD+ID)\(J1+D'*(X-S1-E-N)+(D'*Y1-Y2)/mu);
    Z1=temp;
    
    %updata E
    temp = X-D*Z1-S1-N+Y1/mu;
    par = lambda/mu;
    E1= sign(temp).*max(abs(temp)-par,0);
    
    %update N
    temp=(mu*(X-D*Z1-S1-E1)+Y1)/(2*lambda+mu);
    N1=temp;

    %updata Y1 Y2 Y3
    RES=X-(D*Z1)-S1-E1-N1;
    Y1=Y1+mu*(RES);
    Y2=Y2+mu*(Z1-J1);
    Y3=Y3+mu*(S1-L1);
    %updata mu
    mu=min(mu_max,rou*mu);
    Z=Z1;
    J=J1;
    L=L1;
    S=S1;
    E=E1;
    N=N1;

    s1=norm(X-D*Z-S-E-N,Inf);
    s2=norm(Z-J,Inf);
    s3=norm(S-L,Inf);
    sm=[s1 s2 s3];
    sc=max(sm);
    if iter==1 || mod(iter,50)==0 || sc<epsilon
%     if display
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.3e'), ...
             ',stopC=' num2str(sc,'%2.3e') ]);
    end
    if sc < epsilon
        break
    end
end


end

function Y=svd_threshold(X,r)
[U,S,V] = svd(X, 'econ'); % stable 
 sigma = diag(S);
 svp = length(find(sigma>r));
    if svp>=1
        sigma = sigma(1:svp)-r;
    else
        svp = 1;
        sigma = 0;
    end
    Y = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
end