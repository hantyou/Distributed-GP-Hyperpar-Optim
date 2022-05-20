function agentsPredictionPlot(Agents,mean,var,reso_x,reso_y,range_x1,range_x2,agentsPosiY,...
    fname0,method,eps_export,png_export,visible,fig_export_pix,temp_data,region,contourFlag)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
M=Agents.M;
gcf=figure('visible',visible);
ts_1=linspace(range_x1(1),range_x1(2),reso_x);
ts_2=linspace(range_x2(1),range_x2(2),reso_y);
[mesh_x,mesh_y]=meshgrid(ts_1,ts_2);

tiledlayout(2,M,'TileSpacing','Compact','Padding','Compact');
for m=1:M
    Mean=reshape(mean(m,:),reso_x,reso_y);
    Var=reshape(var(m,:),reso_x,reso_y);
    
    axm=nexttile(m);
    
    surf(mesh_x,mesh_y,(Mean),'edgecolor','none','FaceAlpha',0.9);
    hold on,
    ax = gca;
    ax.YDir = 'normal';
    scatter3(Agents(m).X(1,:),Agents(m).X(2,:),Agents(m).Z,'*')
    scatter3(Agents(m).Position(1),Agents(m).Position(1),agentsPosiY(m)+1,'k^','filled')
    % hold off; xlabel('x1'), ylabel('x2'), zlabel('y'), title(strcat(method,' GPR result - mean'));
    hold off; xlabel('x1'), ylabel('x2'), zlabel('y'), title('mean');
    xlim([range_x1(1),range_x1(2)]);
    ylim([range_x2(1),range_x2(2)]);
	caxis(axm,[6,18]);
    if m==M
        colorbar;
    end
    view(0,90);
    %         subplot(2,M,m+M),
    nexttile(m+M)
    
    surf(mesh_x,mesh_y,(Var)/1,'edgecolor','none','FaceAlpha',0.9);
    hold on,
    ax = gca;
    ax.YDir = 'normal';
    %         Z_ps=interp2(mesh_x,mesh_y,(Var),X(1,:),X(2,:));
    Z_ps=interp2(mesh_x,mesh_y,(Var)/1,Agents(m).X(1,:),Agents(m).X(2,:));
    scatter3(Agents(m).X(1,:),Agents(m).X(2,:),Z_ps,'*');
    set(gca,'ZScale','log')
    hold off;
    xlabel('x1'), ylabel('x2'), zlabel('y')
    xlim([range_x1(1),range_x1(2)]);
    ylim([range_x2(1),range_x2(2)]);
    zlim(10.^[-4.1,2.9])
    zticks(10.^(-4:2:2));
    % title({strcat(method,' GPR result'),'variance (in log plot)'})
    title({'variance (in log plot)'})
    if m==M
        colorbar;
    end
    %view(0,90);
end

sname='DEC predict plot';
s=hgexport('readstyle',sname);
s.Resolution=fig_export_pix;
s.Width=26/8*M;
if eps_export==1
    s.Format='eps';
    fname=fname0;
    if temp_data==3
        fname=strcat(fname,'_',region);
    end
    fname=strcat(fname,'.',s.Format);
    hgexport(gcf,fname,s);
    disp("eps file saved")
end

if png_export==1
    s.Format='png';
    fname=fname0;
    if temp_data==3
        fname=strcat(fname,'_',region);
    end
    fname=strcat(fname,'.',s.Format);
    hgexport(gcf,fname,s);
    disp("png file saved")
    %     pause(1)
end

fname=fname0;
saveas(gcf,fname,'fig');
saveas(gcf,strcat(fname,'_direct_save'),'png');
close(gcf)

if contourFlag==1
    gcf=figure('visible',visible);
    ts_1=linspace(range_x1(1),range_x1(2),reso_x);
    ts_2=linspace(range_x2(1),range_x2(2),reso_y);
    [mesh_x,mesh_y]=meshgrid(ts_1,ts_2);
    
    tiledlayout(2,M,'TileSpacing','Compact','Padding','Compact');
    for m=1:M
        Mean=reshape(mean(m,:),reso_x,reso_y);
        Var=reshape(var(m,:),reso_x,reso_y);
        
        nexttile(m)
        
%         surf(mesh_x,mesh_y,(Mean),'edgecolor','none','FaceAlpha',0.9);
        imagesc(mesh_x(1,:),mesh_y(:,1),Mean);
        hold on,
        contour(mesh_x,mesh_y,Mean,'-r','LineWidth',0.5);
        ax = gca;
        ax.YDir = 'normal';
        scatter(Agents(m).X(1,:),Agents(m).X(2,:),'*')
        scatter(Agents(m).Position(1),Agents(m).Position(2),'k^','filled')
        
        hold off; xlabel('x1'), ylabel('x2'), title(strcat(method,' GPR result - mean'));
        xlim([range_x1(1),range_x1(2)]);
        ylim([range_x2(1),range_x2(2)]);
        if m==M
            colorbar;
        end
        %         subplot(2,M,m+M),
        nexttile(m+M)
        
        surf(mesh_x,mesh_y,(Var)/1,'edgecolor','none','FaceAlpha',0.9);
        hold on,
        ax = gca;
        ax.YDir = 'normal';
        %         Z_ps=interp2(mesh_x,mesh_y,(Var),X(1,:),X(2,:));
        Z_ps=interp2(mesh_x,mesh_y,(Var)/1,Agents(m).X(1,:),Agents(m).X(2,:));
        scatter3(Agents(m).X(1,:),Agents(m).X(2,:),Z_ps,'*');
        set(gca,'ZScale','log')
        hold off;
        xlabel('x1'), ylabel('x2'), zlabel('y')
        xlim([range_x1(1),range_x1(2)]);
        ylim([range_x2(1),range_x2(2)]);
        zlim(10.^[-4.1,2.9])
        zticks(10.^(-4:2:2));
        title({strcat(method,' GPR result'),'variance (in log plot)'})
        if m==M
            colorbar;
        end
    end
    
    sname='DEC predict plot';
    s=hgexport('readstyle',sname);
    s.Resolution=fig_export_pix;
    s.Width=26/8*M;
    if eps_export==1
        s.Format='eps';
        fname=fname0;
        fname=strcat(fname,'_contour');
        if temp_data==3
            fname=strcat(fname,'_',region);
        end
        fname=strcat(fname,'.',s.Format);
        hgexport(gcf,fname,s);
        disp("eps file saved")
    end
    
    if png_export==1
        s.Format='png';
        fname=fname0;
        fname=strcat(fname,'_contour');
        if temp_data==3
            fname=strcat(fname,'_',region);
        end
        fname=strcat(fname,'.',s.Format);
        hgexport(gcf,fname,s);
        disp("png file saved")
        %     pause(1)
    end
    
    fname=fname0;
    fname=strcat(fname,'_contour');
    saveas(gcf,fname,'fig');
    saveas(gcf,strcat(fname,'_direct_save'),'png');
    close(gcf)
end


end

