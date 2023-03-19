function [packedOutput,newRange]=loadRealDataset(packedFieldInfo,ncFname,r_select,lon_lat,where)
disp('Reading Sea Surface Temperature (SST) dataset')
if nargin<2
    ncFname="./SSTdata/20220402090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc";
else
end
%  packedFieldInfo={M,TotalNumLevel,everyAgentsSampleNum,Agents_measure_range,samplingMethod};
M=packedFieldInfo{1};
TotalNumLevel=packedFieldInfo{2};
everyAgentsSampleNum=packedFieldInfo{3};
Agents_measure_range=packedFieldInfo{4};
samplingMethod=packedFieldInfo{5};  % 1. uniformly distirbuted accross region; 2. near agents position, could lose some points if out of range

onlyReadNeccessaryData=1; % whether some extra data are read
readMask=0; % whether reading a mask telling the property of cells
[latitudes,longitude,seaSurfaceTemperature,error]=SSTdataRead(ncFname); % read SST data
seaSurfaceTemperature=seaSurfaceTemperature-273; % transfer kelvin to centigrade
latitudes=flip(latitudes,2);
%% decide four predefined regions
if nargin<5
    if nargin<4
        lon=[145,150.5;
            -93.2,-85.4;
            -50.4,-43.6;
            -61.5,-51.7];
        lat=[37.0,40;
            21.9,28.5
            39.5,45.9;
            -50,-40.1];
        where=["Japan";
            "Caribbean";
            "North Atlantic";
            "Argentina"]; %indicate where are these maps from
    else
        lon=lon_lat(1,:);
        lat=lon_lat(2,:);
        where="UserDefinedRegion";
        r_select=1;
    end
else
end
R=size(lon,1); %total number of Regions
r_select=min(R,r_select);
Xs=cell(R,1);
Zs=cell(R,1);
Errs=cell(R,1);
%% Crop regions
for r=1:R
    lat_idx=find(latitudes>=lat(r,1)&latitudes<=lat(r,2));
    lon_idx=find(longitude>=lon(r,1)&longitude<=lon(r,2));
    Zs{r}=seaSurfaceTemperature(lat_idx,lon_idx);
    Errs{r}=error(lat_idx,lon_idx);
end
%% according to r_select, select the region for simulation
region=where(r_select);
F_true=Zs{r_select};
range_x1=lon(r_select,:);
range_x2=lat(r_select,:);
range=[range_x1;range_x2];
[reso_m,reso_n]=size(F_true);
newRange={range_x1,range_x2,reso_m,reso_n};
reso=[reso_m,reso_n];
Err_sigma=Errs{r_select};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Decide sample points
[mesh_x1,mesh_x2]=meshgrid(linspace(range_x1(1),range_x1(2),reso_n),linspace(range_x2(2),range_x2(1),reso_m));
subSize=ones(M,1)*everyAgentsSampleNum;

L_x1=range_x1(2)-range_x1(1);
L_x2=range_x2(2)-range_x2(1);
Agents_Posi=[unifrnd(-L_x1/2,L_x1/2,1,M)*0.8+(range_x1(1)+range_x1(2))/2;
    unifrnd(-L_x2/2,L_x2/2,1,M)*0.8+(range_x2(1)+range_x2(2))/2];
[X,subSize,sampleIdx] = decideSamplePoints(samplingMethod,subSize,range,Agents_Posi,Agents_measure_range);
sampleSize=sum(subSize);
X1=X(1,:);
X2=X(2,:);

sigma_n=0.45;
ErrorGeneratingMethod=2; % 1.use user-defined noise; 2. use dataset-defined noise
if ErrorGeneratingMethod==1
    sampleError=randn(1,sampleSize)*sigma_n;
elseif ErrorGeneratingMethod==2
    Err_sigma_interp=interp2(mesh_x1,mesh_x2,Err_sigma,X1,X2,'nearest');
    sampleError=normrnd(zeros(1,sampleSize),Err_sigma_interp);
    sigma_n=mean(Err_sigma_interp);
end
%% Take sample
Y=interp2(mesh_x1,mesh_x2,F_true,X1,X2);
agentsPosiY=interp2(mesh_x1,mesh_x2,F_true,Agents_Posi(1,:),Agents_Posi(2,:));
Z=Y+sampleError; % The observation model (measurement model)

packedOutput={X,Y,Z,subSize,sampleIdx,sigma_n,agentsPosiY,F_true,Agents_Posi};

%% plot area
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
saveplot=0;
if saveplot==1
    gcf=figure;
    mkdir("./results")
    tiledlayout(1,ceil(R),'TileSpacing','Compact','Padding','Compact');
    for r=1:R
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

    close gcf;

    gcf=figure('visible','off');
    imagesc(longitude,latitudes,seaSurfaceTemperature);
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
    close gcf;
end

end

