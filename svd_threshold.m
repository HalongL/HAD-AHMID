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