%%
addpath(genpath('./data'));

load urban_162band
hsi=urban_detection;
hsi_gt=groundtruth;

data=hsi;
mask=hsi_gt;
%%
img=data(:,:,[37,18,8]);
%  img=data(:,:,[180,80,10]);

for i=1:3
    max_f=max(max(img(:,:,i)));
    min_f=min(min(img(:,:,i)));
    img(:,:,i)=(img(:,:,i)-min_f)/(max_f-min_f);
end
figure,imshow(img);
figure,imshow(mask,[]);
DataTest=data;
[H,W,Dim]=size(DataTest);
num=H*W;
for i=1:Dim
    DataTest(:,:,i) = (DataTest(:,:,i)-min(min(DataTest(:,:,i)))) / (max(max(DataTest(:,:,i))-min(min(DataTest(:,:,i)))));
end
%%
mask_reshape = reshape(mask, 1, num);
anomaly_map = logical(double(mask_reshape)>0);
normal_map = logical(double(mask_reshape)==0);
Y=reshape(DataTest, num, Dim)';
%%
%AHMID Layer 0
Dict=ConstructionDict(Y,5,8);%LRASR method
alpha=1;
lambda=0.1;
tic
[Z,S,E,N]=AHMID(Y,Dict,alpha,lambda,1); %S-Model
t_layer0=toc;

u_s=mean(S);
S_0=S-u_s;
S_0=sum(S_0.^2,1);

r10=S_0;

% r10=sqrt(sum(S.^2,1));

r10_max = max(r10(:));
taus = linspace(0, r10_max, 5000);
PF10=zeros(1,5000); 
PD10=zeros(1,5000);
for index1 = 1:length(taus)
  tau = taus(index1);
  anomaly_map_rx = (r10> tau);
  PF10(index1) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PD10(index1) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
area_Layer_0 = sum((PF10(1:end-1)-PF10(2:end)).*(PD10(2:end)+PD10(1:end-1))/2);
f_show_0=reshape(r10,[H,W]);
f_show_0=(f_show_0-min(f_show_0(:)))/(max(f_show_0(:))-min(f_show_0(:)));
figure('name','Layer_0'), imshow(f_show_0);
imshow(f_show_0);

%% LAYER1
alpha1=1;
lambda1=1;
tic
[~,Dict1,~,~]=UpDict(Y,S,Dict,Z,alpha1,lambda1,1);%D-Model
lambda1=0.1;
[Z1,S1,E1,N1]=AHMID(Y,Dict1,alpha1,lambda1,1);%S-Model
t_layer1=toc;

u_s1=mean(S1);
S1_1=S1-u_s1;
S1_1=sum(S1_1.^2,1);

% rn_new1=sqrt(sum(S1.^2,1));
rn_new1=S1_1;
rn_max1 = max(rn_new1(:));
taus = linspace(0, rn_max1, 5000);
PFn1=zeros(1,5000);
PDn1=zeros(1,5000);
for index1 = 1:length(taus)
  tau = taus(index1);
  anomaly_map_rx = (rn_new1> tau);
  PFn1(index1) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PDn1(index1) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
area_Layer_1 = sum((PFn1(1:end-1)-PFn1(2:end)).*(PDn1(2:end)+PDn1(1:end-1))/2);
f_show_1=reshape(rn_new1,[H,W]);
f_show_1=(f_show_1-min(f_show_1(:)))/(max(f_show_1(:))-min(f_show_1(:)));
figure('name','Layer_1'), imshow(f_show_1);
imshow(f_show_1);
%% LAYER n ...
