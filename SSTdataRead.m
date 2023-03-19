function [latitudes,longitude,seaSurfaceTemperature,error,mask,ExtraData]=...
    SSTdataRead(ncFname,onlyReadNeccessaryData,readMask)
%%
if nargin<3
    if nargin<2
        if nargin<1
            ncFname="./SSTdata/20220402090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc";
        else
            onlyReadNeccessaryData=1; % whether some extra data are read
            readMask=0; % whether reading a mask telling the property of cells
        end
    else
        readMask=0; % whether reading a mask telling the property of cells
    end
else
end
%% Read necessary data
nci=ncinfo(ncFname);
changeTypeToSingle=1; % whether transfer original data type to single (saves memory)

varNum=length(nci.Variables); % number of variables to be read

latitudes=readVar(ncFname,'lat',changeTypeToSingle);
disp('latitudes read')

longitude=readVar(ncFname,'lon',changeTypeToSingle);
disp('longitude read')

seaSurfaceTemperature=readVar(ncFname,'analysed_sst',changeTypeToSingle);
disp('temperature read')


error=readVar(ncFname,'analysis_error',changeTypeToSingle);
% error(isnan(error))=0;
disp('error read')
%% if mask is needed, read it
if readMask==1
    mask=readVar(ncFname,'mask',changeTypeToSingle);
    oceanArea=mask;
    oceanArea(oceanArea~=1)=0;
    disp('mask read')
    % 1 for ocean
    % 2 for land
    % 5 for open lake
    % 9 for open sea with ice
    % 13 for open lake with ice
end

%% If some unneccessary extra data are needed, read them
ExtraData=cell(1,1);
if onlyReadNeccessaryData==0
    seaIceFrac=readVar(ncFname,'sea_ice_fraction',changeTypeToSingle);
    seaIceFrac(isnan(seaIceFrac))=0;
    disp('IceFrac read')
    i=1;
    ExtraData{i}=seaIceFrac;
    try
        dt_1km_data=readVar(ncFname,'dt_1km_data',changeTypeToSingle);
        dt_1km_data(isnan(dt_1km_data))=0;
        disp('dt_1km_data read')
        i=i+1;
        ExtraData{i}=dt_1km_data;
    catch

    end
    sst_anom=readVar(ncFname,'sst_anomaly',changeTypeToSingle);
    i=i+1;
    ExtraData{i}=sst_anom;
    disp('sst_anom read')
end

%% If needed, store images of the datasets
storeImage=0;
if storeImage==1
    set(0,'DefaultFigureVisible','off')

    sname='MyDefault';
    s=hgexport('readstyle',sname);
    s.Resolution=600;
    s.Format='png';

    gcf=figure;
    imshow(seaSurfaceTemperature,[]);
    colormap('jet');
    fname=strcat('seaSurfaceTemperature_',num2str(s.Resolution),'.',s.Format);
    hgexport(gcf,fname,s);
    disp("seaSurfaceTemperature png file saved")

    gcf=figure;
    imshow(error,[]);
    colormap('jet');
    fname=strcat('error_',num2str(s.Resolution),'.',s.Format);
    hgexport(gcf,fname,s);
    disp("error png file saved")

    if readMask==1
        gcf=figure;
        imshow(mask,[]);
        colormap('jet');
        fname=strcat('mask_',num2str(s.Resolution),'.',s.Format);
        hgexport(gcf,fname,s);
        disp("mask png file saved");
    end

    if onlyReadNeccessaryData==0

        gcf=figure;
        imshow(seaIceFrac,[]);
        colormap('jet');
        fname=strcat('seaIceFrac_',num2str(s.Resolution),'.',s.Format);
        hgexport(gcf,fname,s);
        disp("seaIceFrac png file saved")

        gcf=figure;
        imshow(sst_anom,[]);
        colormap('jet');
        fname=strcat('sst_anom_',num2str(s.Resolution),'.',s.Format);
        hgexport(gcf,fname,s);
        disp("sst_anom png file saved")

        try
            gcf=figure;
            imshow(dt_1km_data,[]);
            colormap('jet');
            fname=strcat('dt_1km_data_',num2str(s.Resolution),'.',s.Format);
            hgexport(gcf,fname,s);
            disp("dt_1km_data png file saved")
        catch
        end
    end
end
end


%% Functions needed
function output=readVar(ncFname,varname,single)
output=ncread(ncFname,varname);
output=flip(output',1);
if single
    output=cast(output,'single');
end
end
