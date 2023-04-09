function [Z,D,E,N]=UpDict(X,S,D_pre,Z_pre,alpha,lambda,display)

if nargin<5
    display = false;
end
rou=1.2;
tol1= 1e-6;
maxIter = 120;%1e6;
mu=1e-2;
mu_max=1e10;
[dim num] = size(X);
SF_norm=norm(S,'fro');
% D1=D_pre;
a=(SF_norm)^2;
IB=eye(dim);
EB=ones(dim);
B=IB-1/dim*EB;
BF_norm=norm(B,'fro');
b = BF_norm^2;
% a= SF_norm^2;
% k=0;
% [H,W,dim]=size(X);
% num=H*W;
% X=reshape(X, num, Dim)';
%[dim,num] = size(Y);
numD = size(D_pre,2);
% 
Z=Z_pre;
J=zeros(numD,num);
L=zeros(dim,num);
E=zeros(dim,num);
N=zeros(dim,num);
D=zeros(dim,numD);
Y1=zeros(dim,num);
Y2=zeros(numD,num);
Y3=zeros(dim,num);
ID=eye(numD);
iter = 0;
% X_F=norm(X,'fro');


while iter<maxIter
    iter = iter + 1;

    %update J
    temp=Z+Y2/mu;
    temp1=svd_threshold(temp,1/mu);
    J1=temp1;
    
    %updata L
    temp=S+Y3/mu;
    L1=solve_l1l2(temp,1/mu);
    
    %updata D
    temp=((X-S-E-N)+Y1/mu)*Z'/((Z*Z')+(2*alpha*(a+b)/mu)*ID);
    D1=temp;
        
    %update Z
    temp=(D1'*D1+ID)\(J1+D1'*(X-S-E-N)+(D1'*Y1-Y2)/mu);
    Z1=temp;
    
    %updata E
    temp = X-D1*Z1-S-N+Y1/mu;
    par = lambda/mu;
    E1= sign(temp).*max(abs(temp)-par,0);
    
    %update N
    temp=(mu*(X-D1*Z1-S-E1)+Y1)/(2*lambda+mu);
    N1=temp;    
    
    %updata Y
    RES=X-(D1*Z1)-S-E1-N1;
    Y1=Y1+mu*(RES);
    Y2=Y2+mu*(Z1-J1);
    Y3=Y3+mu*(S-L1);
    %updata mu
    mu=min(mu_max,rou*mu);
    Z=Z1;
    J=J1;
    L=L1;
    D=D1;
    E=E1;
    N=N1;
   
    s1=norm(X-D*Z-S-E-N,Inf);
    s2=norm(Z-J,Inf);
    s3=norm(S-L,Inf);
    sm=[s1 s2 s3];
    sc=max(sm);
    if iter==1 || mod(iter,50)==0 || sc<tol1
%    if display
       disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.3e') ...
             ',stopC1=' num2str(sc,'%2.3e') ]);
    end
   if(sc<tol1)
       break;
   end
   
%    if display
%         disp('dictionary updated' );
%    end

end
if display
        disp('dictionary updated' );
end