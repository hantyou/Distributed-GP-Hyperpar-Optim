GT = shaperead("Road_LAeq_16h_London\Road_LAeq_16h_London.shp");
% mapshow(GT);
N=length(GT);
X1=zeros(N,1);
X2=zeros(N,1);
NoiseClass=zeros(N,1);
disp('Data read begin')
parfor n=1:N
    gt=GT(n);
    gtx=gt.X(1:end-1);
    gty=gt.Y(1:end-1);
    mx=mean(gtx);
    my=mean(gty);
    if ~isnan(mx)&&~isnan(my)
        X1(n)=mean(gtx);
        X2(n)=mean(gty);
        
        switch gt.NoiseClass
            case '55.0-59.9'
                NoiseClass(n)=(55+59.9)/2;
%                 NoiseClass(n)=(55+59.9)/2+rand*(59.9-55);
            case '60.0-64.9'
                NoiseClass(n)=(60+64.9)/2;
%                 NoiseClass(n)=(60+64.9)/2+rand*(64.9-60);
            case '65.0-69.9'
                NoiseClass(n)=(65+69.9)/2;
%                 NoiseClass(n)=(65+69.9)/2+rand*(69.9-65);
            case '70.0-74.9'
                NoiseClass(n)=(70+74.9)/2;
%                 NoiseClass(n)=(70+74.9)/2+rand*(74.9-70);
            otherwise
                NoiseClass(n)=75+2.5;
%                 NoiseClass(n)=75+2.5;
        end
    else
        X1(n)=-1;
        X2(n)=-1;
        NoiseClass(n)=-1;
    end
end
idx=find(NoiseClass==-1);
X1(idx)=[];
X2(idx)=[];
NoiseClass(idx)=[];
N=length(X1);

figure, scatter3(X1,X2,NoiseClass)
X=[X1';X2'];
Y=NoiseClass;
sigma_n=2;

save('NoiseField.mat','X','Y');