temp_data=3;
if temp_data==1
    %% city temperatures
    disp('The loaded dataset is temperature of cities')
    load('data/city45data.mat');
    load('data/city45T.mat');
    load('data/smhi_temp_17.mat');
    load('data/smhi17_partition.mat');
    cityNum=M;
    dayLength=40;
    idx=find(coord45(:,1)>=62);
    coord45(idx,:)=[];
    temp_17(:,idx)=[];
    idx=find(coord45(:,2)>=18);
    coord45(idx,:)=[];
    temp_17(:,idx)=[];
    %     coord45(:,1)=coord45(:,1)-mean(coord45(:,1));
    %     coord45(:,2)=coord45(:,2)-mean(coord45(:,2));
    % coord45=coord45/max(max(abs(coord45)));
    % temp_17=temp_17/max(abs(temp_17(:)));
    Data=temp_17;
    % temp_17=temp_17-mean(mean(temp_17));
    X=coord45(1:cityNum,:)';
    temp_17_train=temp_17(1:dayLength,1:cityNum);
    temp_train_op=temp_17((1:dayLength)+1,1:cityNum);
    X_coor=kron(X,ones(1,dayLength));
    X_day=kron(ones(1,cityNum),1:dayLength);
    cityIdx=kron(1:cityNum,ones(1,dayLength));
    temp_train_ip=zeros(1,length(X_day));

    for i=1:length(X_day)
        d=X_day(i);
        c=cityIdx(i);
        temp_train_ip(i)=temp_17(d,c);
    end

    X=[X_coor;X_day;temp_train_ip];
    mix=randperm(cityNum*dayLength);
    X=X(:,mix);
    F_true=reshape(temp_train_op,1,cityNum*dayLength);
    F_true=F_true(:,mix);
    Y=F_true;

    newXs_coor=coord45(1:cityNum,:)';
    test_dayLength=size(temp_17,1)-dayLength;
    newXs_coor=kron(newXs_coor,ones(1,test_dayLength));
    newXs_day=kron(ones(1,cityNum),dayLength+1:dayLength+test_dayLength);
    temp_17_test=temp_17(dayLength+1:end,1:cityNum);
    F_test_true=reshape(temp_17_test,1,cityNum*test_dayLength);
    Y_test=temp_17(dayLength+1:end,1:cityNum)';
    newXs=[newXs_coor;newXs_day];


    subSize=ones(1,M);
    unallocatedCityNum=cityNum-sum(subSize);
    for i=1:unallocatedCityNum
        idx=unidrnd(M,1,1);
        subSize(idx)=subSize(idx)+1;
    end
    subSize=subSize*dayLength;
    sampleSize=sum(subSize);
    Z=Y;
    sampleSize=sum(subSize);
    sampleIdx=cumsum(subSize);
    sampleIdx=[0,sampleIdx];
    sigma_n=0.01;


elseif temp_data==0
%% RSSI dataset
    disp('The loaded dataset is RSSI')
    fileRead=readtable('iBeacon_RSSI_Labeled.csv');
    [N,inputDim]=size(fileRead);
    N=round(N/2);
    coor=zeros(2,N);
    for i=1:N
        txt=fileRead{i,1};
        txt=txt{1};
        coor(2,i)=str2double(txt(2:3));
        coor(1,i)=double(uint8(txt(1))-uint8('A')+1);
    end
    input=fileRead(1:N,3:end);
    input=table2array(input)';
    X=input;
    Y=coor(1,:);

    subSize=ones(1,M);
    unallocatedN=N-sum(subSize);
    for i=1:unallocatedN
        idx=unidrnd(M,1,1);
        subSize(idx)=subSize(idx)+1;
    end

    sampleSize=sum(subSize);
    Z=Y;
    sampleSize=sum(subSize);
    sampleIdx=cumsum(subSize);
    sampleIdx=[0,sampleIdx];
    sigma_n=0.01;
elseif  temp_data==2
%% City road noise level
    disp('The loaded dataset is city noise')

    load('NoiseField.mat');
    N=length(X);
    randIdx=randperm(N);
    newN=1024*2;
    Y=Y';
    X=X(:,randIdx(1:newN));
    Y=Y(:,randIdx(1:newN));
    N=newN;
    subSize=ones(1,M);
    unallocatedN=N-sum(subSize);
    for i=1:unallocatedN
        idx=unidrnd(M,1,1);
        subSize(idx)=subSize(idx)+1;
    end
    range_X1=[min(X(1,:)),max(X(1,:))];
    range_X2=[min(X(2,:)),max(X(2,:))];
    X=100*X./max(abs(X(:)));
    % Y=Y-min(Y);
    sampleSize=sum(subSize);
    Z=Y;
    sigma_n=2;
    Z=Y+sigma_n^2*rand(1,N);
    sampleSize=sum(subSize);
    sampleIdx=cumsum(subSize);
    sampleIdx=[0,sampleIdx];
elseif temp_data==3
    disp('Reading Sea Surface Temperature (SST) dataset')
    SSTdataRead;
    % seaSurfaceTemperature=(seaSurfaceTemperature-273)./(max(seaSurfaceTemperature(:))-min(seaSurfaceTemperature(:)));
    seaSurfaceTemperature=seaSurfaceTemperature-273;
    latitudes=flip(latitudes,2);
%     lon=[143.8,150.5;
%         -93.2,-85.4;
%         -50.4,-43.6;
%         -61.5,-51.7];

    lon=[145,150.5;
    -93.2,-85.4;
    -50.4,-43.6;
    -61.5,-51.7];

%     lat=[36.0,41.9;
%         21.9,28.5
%         39.5,45.9;
%         -50,-40.1];
      lat=[37.0,40;
        21.9,28.5
        39.5,45.9;
        -50,-40.1];  
    where=["Japan";
        "Caribbean";
        "North Atlantic";
        "Argentina"];

    R=4; %total number of Regions
    Xs=cell(R,1);
    Zs=cell(R,1);
    Errs=cell(R,1);
    gcf=figure('visible','off');

    tiledlayout(1,ceil(R),'TileSpacing','Compact','Padding','Compact');
    for r=1:R
        lat_idx=find(latitudes>=lat(r,1)&latitudes<=lat(r,2));
        lon_idx=find(longitude>=lon(r,1)&longitude<=lon(r,2));
        [mesh_lon,mesh_lat]=meshgrid(lon_idx,lat_idx);
        mesh_lon=mesh_lon(:);
        mesh_lat=mesh_lat(:);
        [mesh_size_lat,mesh_size_lon]=size(mesh_lon);
        Zs{r}=seaSurfaceTemperature(lat_idx,lon_idx);
        Errs{r}=error(lat_idx,lon_idx);
%         seaSurfaceTemperature(lat_idx,lon_idx)=0;
        %subplot(2,ceil(R/2),r);
        nexttile(r);
        imshow(Zs{r},[0,30]);
        colormap("jet");
if r==R
        colorbar;
end
        title(where(r));
    end
    sname='MyDefault';
    s=hgexport('readstyle',sname);
    s.Resolution=300;
    s.Format='png';
    s.Width=9.2;
    s.Height=2.3;
    s.ScaledFontSize=100;
    s.FontSizeMin=12;
    s.Bounds='tight';

    hgexport(gcf,"./results/SSTregion/cropped_temperature_region.png",s);
    s.Format='eps';
    hgexport(gcf,"./results/SSTregion/cropped_temperature_region.eps",s);
    
    % saveas(gcf,"./results/SSTregion/cropped_temperature_region",'png');
    close gcf;
    gcf=figure('visible','off');
    
    imagesc(longitude,latitudes,seaSurfaceTemperature);
%     imshow(seaSurfaceTemperature,[]);
    hold on;
    for r=1:R
        rectangle('Position',[lon(r,1),lat(r,1),lon(r,2)-lon(r,1),lat(r,2)-lat(r,1)],'EdgeColor','m','LineWidth',2);
        text(mean(lon(r,:)),mean(lat(r,:)),where(r),'Color','black','FontSize',12);
    end
    xlabel('longitude')
    ylabel('latitude')
    colormap("jet")
    colorbar;
    set(gca, 'YDir','normal');
    gcf.Units='Inches';
    gcf.Position(3)=18.5*0.5;
    gcf.Position(4)=gcf.Position(3)/19*9;
    title('Global SST and selected areas')
    sname='MyDefault';
    s=hgexport('readstyle',sname);
    s.Resolution=300;
    s.Format='png';
    s.ScaledFontSize=100;
    s.FontSizeMin=12;
    s.Bounds='tight';
    
    hgexport(gcf,"./results/SSTregion/where_are_these_regions.png",s);
    s.Format='eps';
    hgexport(gcf,"./results/SSTregion/where_are_these_regions.eps",s);
%     exportgraphics(gcf,'./results/SSTregion/where are these regions.png','Resolution',300);
%     saveas(gcf,"./results/SSTregion/where_are_these_regions",'png');
    close gcf;

    r_select=1;
    region=where(r_select);
    F_true=Zs{r_select};
    range_x1=lon(r_select,:);
    range_x2=lat(r_select,:);
    range=[range_x1;range_x2];
    [reso_m,reso_n]=size(F_true);
    reso=[reso_m,reso_n];
    Err_sigma=Errs{r_select};
    
    %resample
        [mesh_x1,mesh_x2]=meshgrid(linspace(range_x1(1),range_x1(2),reso_n),linspace(range_x2(2),range_x2(1),reso_m));

    %% Decide sample points
    samplingMethod=2; % 1. uniformly distirbuted accross region; 2. near agents position, could lose some points if out of range
    subSize=ones(M,1)*everyAgentsSampleNum;
%     Agents_Posi=[unifrnd(range_x1(1),range_x1(2),1,M)*0.8;
%         unifrnd(range_x2(1),range_x2(2),1,M)*0.8];
    L_x1=range_x1(2)-range_x1(1);
    L_x2=range_x2(2)-range_x2(1);
    Agents_Posi=[unifrnd(-L_x1/2,L_x1/2,1,M)*0.8+(range_x1(1)+range_x1(2))/2;
        unifrnd(-L_x2/2,L_x2/2,1,M)*0.8+(range_x2(1)+range_x2(2))/2];
    [X,subSize,sampleIdx] = decideSamplePoints(samplingMethod,subSize,range,Agents_Posi,Agents_measure_range);
    sampleSize=sum(subSize);
    X1=X(1,:);
    X2=X(2,:);

    sigma_n=0.45;
    ErrorGeneratingMethod=2;
    if ErrorGeneratingMethod==1
        sampleError=randn(1,sampleSize)*sigma_n;
    elseif ErrorGeneratingMethod==2
        [e_m,e_n]=size(Err_sigma);
        Err_sigma_interp=interp2(mesh_x1,mesh_x2,Err_sigma,X1,X2,'nearest');
        sampleError=normrnd(zeros(1,sampleSize),Err_sigma_interp);
        sigma_n=mean(Err_sigma_interp);
    end
    %% Take sample
    Y=interp2(mesh_x1,mesh_x2,F_true,X1,X2);
    agentsPosiY=interp2(mesh_x1,mesh_x2,F_true,Agents_Posi(1,:),Agents_Posi(2,:));
    Z=Y+sampleError; % The observation model (measurement model)
    %% Clear Useless vars
    clear latitudes longitude mesh_lat mesh_lon s nci seaSurfaceTemperature error;

end
