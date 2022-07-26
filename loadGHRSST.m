function [F_true,range,reso] = loadGHRSST(regionSelect,custom_range_lon,custome_range_lat)
%LOADGHRSST Summary of this function goes here
%   If you want user defined reange, make regionSelect = 0
%% Input Interface
switch nargin
    case 1
        disp('Choose a pre defined region.')
    otherwise
        disp('Choose a user defined region.')
end
%% Data read
disp('Reading Sea Surface Temperature (SST) dataset')
SSTdataRead;
% seaSurfaceTemperature=(seaSurfaceTemperature-273)./(max(seaSurfaceTemperature(:))-min(seaSurfaceTemperature(:)));
seaSurfaceTemperature=seaSurfaceTemperature-273;
latitudes=flip(latitudes,2);

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
    "Argentina"];

[R,~]=size(lon); %total number of Regions
Xs=cell(R,1);
Zs=cell(R,1);
Errs=cell(R,1);
for r=1:R
    lat_idx=find(latitudes>=lat(r,1)&latitudes<=lat(r,2));
    lon_idx=find(longitude>=lon(r,1)&longitude<=lon(r,2));
    [mesh_lon,mesh_lat]=meshgrid(lon_idx,lat_idx);
    mesh_lon=mesh_lon(:);
    mesh_lat=mesh_lat(:);
    [mesh_size_lat,mesh_size_lon]=size(mesh_lon);
    Zs{r}=seaSurfaceTemperature(lat_idx,lon_idx);
    Errs{r}=error(lat_idx,lon_idx);
end
r_select=regionSelect;
region=where(r_select);
F_true=Zs{r_select};
range_x1=lon(r_select,:);
range_x2=lat(r_select,:);
range=[range_x1;range_x2];
[reso_m,reso_n]=size(F_true);
reso=[reso_m,reso_n];

%% Figure output
if 0
    gcf=figure('visible','off');
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
end
%%

end

