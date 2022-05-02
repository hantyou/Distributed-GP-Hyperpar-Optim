

ncFname="./SSTdata/20220402090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc";

onlyReadNeccessaryData=1;
readMask=0;

nci=ncinfo(ncFname);
changeTypeToSingle=1;

varNum=length(nci.Variables);

latitudes=readVar(ncFname,'lat',changeTypeToSingle);
disp('latitudes read')

longitude=readVar(ncFname,'lon',changeTypeToSingle);
disp('longitude read')

seaSurfaceTemperature=readVar(ncFname,'analysed_sst',changeTypeToSingle);
disp('temperature read')


error=readVar(ncFname,'analysis_error',changeTypeToSingle);
% error(isnan(error))=0;
disp('error read')

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

if onlyReadNeccessaryData==0
    
    seaIceFrac=readVar(ncFname,'sea_ice_fraction',changeTypeToSingle);
    seaIceFrac(isnan(seaIceFrac))=0;
    disp('IceFrac read')
    
    try
        dt_1km_data=readVar(ncFname,'dt_1km_data',changeTypeToSingle);
        dt_1km_data(isnan(dt_1km_data))=0;
        disp('dt_1km_data read')
    catch
        
    end
    
    sst_anom=readVar(ncFname,'sst_anomaly',changeTypeToSingle);
    disp('sst_anom read')
    
end

%%
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
function output=readVar(ncFname,varname,single)
output=ncread(ncFname,varname);
output=flip(output',1);
if single
    output=cast(output,'single');
end
end
