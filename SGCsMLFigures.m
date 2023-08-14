% written by Peizhe Wu at 1/27/2023 to consolidate the codes for ML paper
% This code includes two main parts, 
% part I: analyze the raw machine learning data from python, calculate the
% cochlear frequency, and bin it. 
% part II: plot the figures for the paper. 
% database version: ('MLpaper2022data.mat')
% the codes for machine learning are written in python 
clear
cd('\\apollo\research\ENT\Liberman\Peizhe Wu\06 Publications\Work in progress\8 SGC technique paper\01 Code and data for figures v12')
figurepath = '\\apollo\research\ENT\Liberman\Peizhe Wu\06 Publications\Work in progress\8 SGC technique paper\01 Code and data for figures v12';
load('MLpaper2022data.mat')
cd (figurepath)

%% Plot the transformed trace vs the originial trace
% [406,547,588,612,613,699,704,750,791,829,888,908,977,1046,1047,1112,1113,1119,1120,1286,1301,1314,1331,1335,1346,1392]
for trayN = [1191,1424]
dir = 'D:\SGC project data\Human temporal bone 3D stack\Tray';
dir2 = '\\apollo\research\ENT\Liberman\Peizhe Wu\01 Human study WPZ\01 Human Ear Raw Data\Human temporal bone 3D stack\Tray';
OCPath = dir + string(trayN) +'\xyz coordinates for HC v6.xlsx';
Mv6 = readtable(OCPath);
OCPath = dir + string(trayN) +'\xyz coordinates for HC.xlsx';
Mv1 = readtable(OCPath);

RCPath = dir + string(trayN) +'\xyz coordinates for RC v6.xlsx';
MRv6 = readtable(RCPath);
RCPath = dir + string(trayN) +'\xyz coordinates for RC.xlsx';
MRv1 = readtable(RCPath);

SGCPath = dir + string(trayN) +'\xyz coordinates for SGN v6.xlsx';
MSGCv6 = readtable(SGCPath);
SGCPath = dir + string(trayN) +'\xyz coordinates for SGN.xlsx';
MSGCv1 = readtable(SGCPath);

figure('Position',[599 303 1109 750]) 
hold on 
OCX = Mv1.xcor*0.503/1000;
OCY = Mv1.ycor*0.503/1000;
OCZ = Mv1.z_HC;
plot3(OCX,OCY,OCZ,'k--','linewidth',2)

OCX = Mv6.xcor*0.503/1000;
OCY = Mv6.ycor*0.503/1000;
OCZ = Mv6.Zcor;
plot3(OCX,OCY,OCZ,'k','linewidth',2)
% smooth the RC and OC spiral
OCloc = interparc([0:0.001:1],OCX,OCY,OCZ);
for i = 1:3
f = fit([1:1:length(OCloc)]',OCloc(:,i),'smoothingspline','SmoothingParam',0.00002);
OCloc(:,i) = f([1:1:length(OCloc)]');
end
plot3(OCloc(:,1),OCloc(:,2),OCloc(:,3),'r-','linewidth',2)

OCX = MRv1.xcor*0.503/1000;
OCY = MRv1.ycor*0.503/1000;
OCZ = MRv1.z_SGN;
plot3(OCX,OCY,OCZ,'b--','linewidth',2)

OCX = MRv6.xcor*0.503/1000;
OCY = MRv6.ycor*0.503/1000;
OCZ = MRv6.Zcor;
plot3(OCX,OCY,OCZ,'g--','linewidth',2)
OCloc = interparc([0:0.001:1],OCX,OCY,OCZ);
for i = 1:3
f = fit([1:1:length(OCloc)]',OCloc(:,i),'smoothingspline','SmoothingParam',0.00002);
OCloc(:,i) = f([1:1:length(OCloc)]');
end
plot3(OCloc(:,1),OCloc(:,2),OCloc(:,3),'g','linewidth',2)

SGCX = MSGCv6.xcor*0.503/1000;
SGCY = MSGCv6.ycor*0.503/1000;
SGCZ = MSGCv6.Zcor;
scatter3(SGCX,SGCY,SGCZ,5,'r','linewidth',2)

SGCX = MSGCv1.xcor*0.503/1000;
SGCY = MSGCv1.ycor*0.503/1000;
SGCZ = MSGCv1.z_SGN;
scatter3(SGCX,SGCY,SGCZ,5,'k')

OCloc = interparc([0:0.001:1],OCX,OCY,OCZ);

legend({'HC trace original','HC trace aftercorrection','smoothed HC trace',...
    'RC trace original','RC trace after correction','smoothed RC trace',...
    'SGC coordinate after correction','SGC coordinate original'})
title(['tray',num2str(trayN)])

xlim([0,10]);ylim([0,10]);zlim([0,10])
saveas(gcf,dir + string(trayN) +'\'+'Tray'+string(trayN)+'_Original vs corrected coordinate.fig')
saveas(gcf,dir2 + string(trayN) +'\'+'Tray'+string(trayN)+'_Original vs corrected coordinate.fig')

end

%% part I, smooth the 3D reconstruction a calculate cochlear frequencies, bin the data 
%% part I - 1: figure out the individual transformation function from SGN to HC 
% extract data from spreadsheet, calculate the SGC map
n = 0;TRAYN = [];IHCpercent = [];RCpercent = [];
for collumnN =  2:175

        if isnumeric(database{80,collumnN}) &~isempty(database{80,collumnN})
            trayN = database{80,collumnN};
            dir = 'D:\SGC project data\Human temporal bone 3D stack\Tray';
            OCPath = dir + string(trayN) +'\xyz coordinates for HC v6.xlsx';
            RCPath = dir + string(trayN) +'\xyz coordinates for RC v6.xlsx';
            SGNpath = dir + string(trayN) +'\xyz coordinates for SGN v6.xlsx';
            if any(truefiles(:) == trayN)
            M = readtable(OCPath);database{84,collumnN} = M;
            OCX =M.xcor*0.503/1000;OCY = M.ycor*0.503/1000;OCZ = M.Zcor;
            OCloc = interparc([0:0.001:1],OCX,OCY,OCZ);
            
            M = readtable(RCPath);database{83,collumnN} = M;
            RCX =M.xcor*0.503/1000;RCY = M.ycor*0.503/1000;RCZ = M.Zcor;
            RCloc = interparc([0:0.001:1],RCX,RCY,RCZ);
%             RCloc = interparc([0:0.001:1],RCloc(1:970,1),RCloc(1:970,2),RCloc(1:970,3));

            M = readtable(SGNpath);database{82,collumnN} = M;
            SGNX =M.xcor*0.503/1000;SGNY = M.ycor*0.503/1000;SGNZ =  M.Zcor;
            % smooth the RC and OC spiral
            for i = 1:3
            f = fit([1:1:1001]',OCloc(:,i),'smoothingspline','SmoothingParam',0.00002);
            OCloc(:,i) = f([1:1:1001]');
            f = fit([1:1:1001]',RCloc(:,i),'smoothingspline','SmoothingParam',0.00002);
            RCloc(:,i) = f([1:1:1001]');
            end
            % Map Rosenthal's canal to Organ of Corti
            clear dist index
            for i = 1:1000
                OC = OCloc(i,:);
                dist(i) = 1000;
                for k = 1:1000
                    if sqrt((OC(1)-RCloc(k,1)).^2 +(OC(2)-RCloc(k,2)).^2+(OC(3)-RCloc(k,3)).^2) < dist(i)
                        dist(i) = sqrt((OC(1)-RCloc(k,1)).^2 +(OC(2)-RCloc(k,2)).^2+(OC(3)-RCloc(k,3)).^2);
                        index(i) = k;
                    end
                end
            end  
            index(min(find(index == max(index))):end) = 1000;
            Smoothedindex = floor(smoothdata(index,'rlowess',150));
      
            % Map SGN location to Rosenthal's canal 
            clear index  dist
            for i = 1:length(SGNX)
                dist(i) = 1000;
                for k = 1:1000
                    if sqrt((SGNX(i)-RCloc(k,1)).^2+(SGNY(i)-RCloc(k,2)).^2+(SGNZ(i)-RCloc(k,3)).^2) <dist(i)
                        dist(i) = sqrt((SGNX(i)-RCloc(k,1)).^2+(SGNY(i)-RCloc(k,2)).^2+(SGNZ(i)-RCloc(k,3)).^2);
                        index(i) = k;
                    end
                end
            end
             M.RCpercentage= index';
            % Map SGN to Organ of Corti 
            clear backtoOC

            for i = 1: length(index)
                distance = abs(Smoothedindex-index(i));
                backtoOC(i) = nanmean(find(distance == min(distance)));
                
            end
            figure; histogram(backtoOC)
            % write to database 
            M.percentfrombasematchtov6data= backtoOC';
            database{82,collumnN} = M;
            else
                database{82,collumnN} =[];
                database{84,collumnN} =[];
                database{83,collumnN} =[];

                
            end
        end
end

%% Part II
%% Figure 3 plot the manual count vs machine count
% part-I  plot ML count vs Jen's Neurolucida count
cd ([figurepath,'\Figure 3'])
fig = figure('position',[1033 457 750 450])
hold on 
B_1 = [0,0,0]; % young 0-1
B_2 = [0,0,0]+0.45; % 1-50 yrs
B_3 = [0,0,0]+0.65; % 50-75 yrs
B_4 = [0,0,0]+0.9; % 75-100 yrs

opts = spreadsheetImportOptions("NumVariables", 15);
% Specify sheet and range
opts.Sheet = "ML vs Jen Zurek";
opts.DataRange = "B2:P82";
% Specify column names and types
opts.VariableNames = ["SlideTray2166", "Seg", "Jencount", "MLcount", "VarName6", "VarName7", "SlideTray2167", "Seg1", "Jencount1", "MLcount1", "VarName12", "VarName13", "Seg2", "Jencount2", "MLcount2"];
opts.VariableTypes = ["double", "string", "double", "double", "string", "string", "double", "string", "double", "double", "string", "string", "string", "double", "double"];
% Specify variable properties
opts = setvaropts(opts, ["Seg", "VarName6", "VarName7", "Seg1", "VarName12", "VarName13", "Seg2"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["Seg", "VarName6", "VarName7", "Seg1", "VarName12", "VarName13", "Seg2"], "EmptyFieldRule", "auto");
SGNcountsfromallthreemethodsS1 = readtable(string([figurepath,'\Figure 3\SGN counts from all three methods.xlsx']), opts, "UseExcel", false);

xvalue = SGNcountsfromallthreemethodsS1.Jencount2;
yvalue = SGNcountsfromallthreemethodsS1.MLcount2
gscatter(xvalue,yvalue,SGNcountsfromallthreemethodsS1.Seg2,[B_2;B_3;B_4;B_1],'.',25)
fithi = fitlm(xvalue,yvalue)
x = [0:1:250]';[prehi,CIhi]=predict(fithi,x);
plot(x,prehi,'color','r','linestyle','-','linewidth',0.5)
plot(x,CIhi(:,1),'color','r','linestyle','--','linewidth',0.5)
plot(x,CIhi(:,2),'color','r','linestyle','--','linewidth',0.5)

ax = gca;
pbaspect(ax,[5 5 1])
axis([-5 250 -5 250]);
ax.XMinorTick = 'off';ax.TickDir = 'out'
ax.Color = 'none'
ax.FontSize = 10;

print(fig,'-dpdf')

% Figure 3B  plot ML count vs Makary count 
cd ([figurepath,'\Figure 3'])
uiopen([figurepath,'\Figure 3\blind assessment for all the cases tested in ML algorithm.xlsx'],1)
load('National-tb-database.mat')
clear year
for i = 1:length(blindassessmentforallthecasestestedinMLalgorithm.TrayN)
    tray = blindassessmentforallthecasestestedinMLalgorithm.TrayN(i);
    index= max([find(LeftTray ==tray),find(RightTray ==tray)]);
    year (i) = yearofdeath(index);
end
xvalue = blindassessmentforallthecasestestedinMLalgorithm.ClassicCount;
yvalue = blindassessmentforallthecasestestedinMLalgorithm.Machinecount;

fig = figure('position',[0 0 750 450])
hold on 
gscatter(xvalue,yvalue,blindassessmentforallthecasestestedinMLalgorithm.SecondratingThiscolumnisusedinthefigure,[B_4;B_1],'.',25)
fithi = fitlm(xvalue(blindassessmentforallthecasestestedinMLalgorithm.SecondratingThiscolumnisusedinthefigure == 'OK'),...
    yvalue(blindassessmentforallthecasestestedinMLalgorithm.SecondratingThiscolumnisusedinthefigure == 'OK'))
x = [0:1:350]';[prehi,CIhi]=predict(fithi,x);
plot(x,prehi,'color','r','linestyle','-','linewidth',0.5)
plot(x,CIhi(:,1),'color','r','linestyle','--','linewidth',0.5)
plot(x,CIhi(:,2),'color','r','linestyle','--','linewidth',0.5)

ax = gca;
pbaspect(ax,[5 5 1])
axis([-5 350 -5 350]);
ax.XMinorTick = 'off';ax.TickDir = 'out'
ax.Color = 'none'
ax.FontSize = 10;
print(fig,'-dpdf')

%% Figure 4 superimpose Neurolucida count with ML count 
clear Neurolucidacount
cd ([figurepath,'\Figure 4\Correlate Neurolucida with ML result\Results Zurek R\\\EveryNeurolucidafoundSGN'])
copyfile 'NeurolucidacountZurek_rt_201_seg_2_Tray2166Slide201_foreveryNeurolucidafoundSGN.xlsx' 'C:\Users\WUPEIZH\Desktop' f
cd 'C:\Users\WUPEIZH\Desktop' 
movefile 'NeurolucidacountZurek_rt_201_seg_2_Tray2166Slide201_foreveryNeurolucidafoundSGN.xlsx'  'Neurolucidacount.xlsx' f
uiopen(['C:\Users\WUPEIZH\Desktop\Neurolucidacount.xlsx'],1)
NeurolucidaX = Neurolucidacount.Xcor;
NeurolucidaY = Neurolucidacount.Ycor;
figure; 
hold on 
scatter(NeurolucidaX,NeurolucidaY,'r')
title('Neurolucida found SGN')

clear Neurolucidacount
cd ([figurepath,'\Figure 4\Correlate Neurolucida with ML result\Results Zurek R\\\\EveryMLfoundSGN'])
copyfile 'NeurolucidacountZurek_rt_201_seg_2_Tray2166Slide201_foreveryMLfoundSGN.xlsx' 'C:\Users\WUPEIZH\Desktop' f
cd 'C:\Users\WUPEIZH\Desktop' 
movefile 'NeurolucidacountZurek_rt_201_seg_2_Tray2166Slide201_foreveryMLfoundSGN.xlsx'  'Neurolucidacount.xlsx' f
uiopen(['C:\Users\WUPEIZH\Desktop\Neurolucidacount.xlsx'],1)
NeurolucidaX = Neurolucidacount.Xcor;
NeurolucidaY = Neurolucidacount.Ycor;
scatter(NeurolucidaX,NeurolucidaY,'filled','k')
title('Neurolucida found SGN')


%%  Part II - figure 5 C plot the transformation curve 
cd(figurepath)
Smoothedindex = NaN(175,1000);
for columnN = 2:175
if ~isempty(database{84,columnN} )
M = database{84,columnN};
OCX =M.xcor*0.503/1000;OCY = M.ycor*0.503/1000;OCZ = M.Zcor;
OCloc = interparc([0:0.001:1],OCX,OCY,OCZ);
M = database{83,columnN};
RCX =M.xcor*0.503/1000;RCY = M.ycor*0.503/1000;RCZ = M.Zcor;
RCloc = interparc([0:0.001:1],RCX,RCY,RCZ);
clear dist index
for i = 1:1000
OC = OCloc(i,:);
dist(i) = 1000;
for k = 1:1000
    if sqrt((OC(1)-RCloc(k,1)).^2 +(OC(2)-RCloc(k,2)).^2+(OC(3)-RCloc(k,3)).^2) < dist(i)
        dist(i) = sqrt((OC(1)-RCloc(k,1)).^2 +(OC(2)-RCloc(k,2)).^2+(OC(3)-RCloc(k,3)).^2);
        index(i) = k;
    end
end
end
index(min(find(index ==max(index))):end) = 1000;
Smoothedindex(columnN,:) = smoothdata(index,'rlowess',150);
end 
end
figure; hold on 
for i = 2:175
    plot(Smoothedindex(i,:),'color',[0,0,0]+0.7)
end
x = [1:1:1000];y = nanmean(Smoothedindex);z=nanstd(Smoothedindex)/sqrt(length(find(~isnan(Smoothedindex(:,1)))))
ploterrorbar(x(5:20:1000),y(5:20:1000),z(5:20:1000),'k',2)
ylim([-50,1000]);xlim([-50,1000])
set(gca,'xscale','linear','YTick',[0:100:1000],'XTick',[0:100:1000])
ax = gca;pbaspect(ax,[5 5 1])
print(gcf,'-dpdf')

%%  Part II - figure 5 B,D  - plot the 3D recon 
cd(figurepath)
columnN =  93;
M = database{84,columnN};
OCX =M.xcor*0.503/1000;OCY = M.ycor*0.503/1000;OCZ = M.Zcor;
OCloc = interparc([0:0.001:1],OCX,OCY,OCZ);
M = database{83,columnN};
RCX =M.xcor*0.503/1000;RCY = M.ycor*0.503/1000;RCZ = M.Zcor;
RCloc = interparc([0:0.001:1],RCX,RCY,RCZ);
M = database{82,columnN};
SGNX =M.xcor*0.503/1000;SGNY = M.ycor*0.503/1000;SGNZ =  M.Zcor;SGNinnervation = M.percentfrombasematchtov6data;
        
% smooth the RC and OC spiral
for i = 1:3
f = fit([1:1:length(OCloc)]',OCloc(:,i),'smoothingspline','SmoothingParam',0.00002);
OCloc(:,i) = f([1:1:length(OCloc)]');
f = fit([1:1:length(RCloc)]',RCloc(:,i),'smoothingspline','SmoothingParam',0.00002);
RCloc(:,i) = f([1:1:length(RCloc)]');

end
clear dist indexRC
% calculate the transformation function 
for i = 1:1000
OC = OCloc(i,:);dist(i) = 1000;
for k = 1:1000
    if sqrt((OC(1)-RCloc(k,1)).^2 +(OC(2)-RCloc(k,2)).^2+(OC(3)-RCloc(k,3)).^2) < dist(i)
        dist(i) = sqrt((OC(1)-RCloc(k,1)).^2 +(OC(2)-RCloc(k,2)).^2+(OC(3)-RCloc(k,3)).^2);
        indexRC(i) = k;
    end
end
end

indexRC(min(find(indexRC ==1000)):end) = 1000;
Smoothedindex = floor(sort(smoothdata(indexRC,'rlowess',100)));
Smoothedindex(find(Smoothedindex <=0)) = 1;
figure; plot(indexRC)
for i = 1: length(indexRC)
    distance = abs(Smoothedindex-indexRC(i));
    backtoOC(i) = nanmean(find(distance == min(distance)));

end
figure; hold on 
plot3(OCloc(:,1),OCloc(:,2),OCloc(:,3),'r','linewidth',3)
plot3(RCloc(:,1),RCloc(:,2),RCloc(:,3),'k','linewidth',3)
c = jet(1000);colormap( jet(1000)) ;
for i = 10:10:1000
    plot3([OCloc(i,1),RCloc(Smoothedindex(i),1)],[OCloc(i,2),RCloc(Smoothedindex(i),2)],[OCloc(i,3),RCloc(Smoothedindex(i),3)],'color',c(i,:),'linewidth',0.8)
end
for i = 50:100:1000
    scatter3(RCloc(i,1),RCloc(i,2),RCloc(i,3),'k','linewidth',3)
    scatter3(OCloc(i,1),OCloc(i,2),OCloc(i,3),'r','linewidth',3)
end
title(['columN',num2str(columnN)]);
ylim([1,10]);xlim([0,10]);zlim([0,10])
ax = gca;pbaspect(ax,[5 5 5])
clear indexSGN  dist
for i = 1:length(SGNX)
    dist(i) = 1000;
    for k = 1:1000
        if sqrt((SGNX(i)-RCloc(k,1)).^2+(SGNY(i)-RCloc(k,2)).^2+(SGNZ(i)-RCloc(k,3)).^2) <dist(i)
            dist(i) = sqrt((SGNX(i)-RCloc(k,1)).^2+(SGNY(i)-RCloc(k,2)).^2+(SGNZ(i)-RCloc(k,3)).^2);
            indexSGN(i) = k;
        end
    end
end
figure; hold on 
for i =1:1:length(SGNinnervation)
% if SGNinnervation(i) > 100 & SGNinnervation(i) < 300
   scatter3(SGNX(i),SGNY(i),SGNZ(i),5,c(floor(SGNinnervation(i)),:)) % plot the SGNs 
%  end
% uncomment the next two lines to plot the projection lines 
   plot3([SGNX(i),RCloc(indexSGN(i),1)],[SGNY(i),RCloc(indexSGN(i),2)],...
       [SGNZ(i),RCloc(indexSGN(i),3)],'color',c(floor(SGNinnervation(i)),:),'linewidth',0.1)% plot the projection line 
end
ylim([-5,10]);xlim([-5,10]);zlim([-5,10])
ax = gca;pbaspect(ax,[5 5 5])

%% Part II - figure 6 - plot SGN innervation density as function of freuqency 
exclusion = [11,20,21,60,61,62,63,64,65,66,67,68,69,71,72,73,74,75,76,77,78,...
    79,80,95,98,99,102,103,106,120,122,126,130,135,136,137,32,125,138]+1; 
for i = 2:175
    if ~any(i == exclusion)
       exclu(i) = true;
    else
        exclu(i) = false;
    end
end
clear SGN Age ANF
row = 82;EDGES = [0:10:100]
for i = 2:175
    if isnumeric(database{5,i} )& ~isnan(database{5,i})
        Age(i) = database{5,i};
    else
        Age(i) = NaN;
    end
    if isnumeric(database{11,i} )
        history(i) = database{11,i};

    else

        history(i) = NaN;
    end
    if ~isempty(database{row,i})

        SGCfreq = database{row,i}.percentfrombasematchtov6data;
        [NSGNeachbin4,~, ~] = histcounts(SGCfreq./10, EDGES);
        SGN(i,1:length(NSGNeachbin4)) = NSGNeachbin4;
        IHC(i,:) =  database{33,i}(:,1);
        OHC(i,:) =  database{33,i}(:,5);
        for K = 1:5
            ANF(i,K) = database{14+K,i};
        end
        else
        SGN(i,1:length(EDGES)-1) = NaN;
        OHC(i,1:20) = NaN;
        IHC(i,1:20) = NaN;   
        ANF(i,1:5) = NaN;

    end  
end

% index1 = find(Age<= 10& ~isnan(SGN(:,1))' & exclu & history ==0);
index1 = find(Age<= 10& ~isnan(SGN(:,1))' & exclu );
youngmeanSGN = nanmean(SGN(index1,:));
youngsemSGN =  nanstd(SGN(index1,:))/sqrt(length(index1));
Agerange = [10,50];
% index1 = find(Age>Agerange(1) & Age<= Agerange(2)& ~isnan(SGN(:,1))' & exclu & history ==0)
index1 = find(Age>Agerange(1) & Age<= Agerange(2)& ~isnan(SGN(:,1))' & exclu );
youngadultmeanSGN = nanmean(SGN(index1,:)) ;
youngadultsemSGN =  nanstd(SGN(index1,:))/sqrt(length(index1));
Agerange = [50,75];
% index1 = find(Age>Agerange(1) & Age<= Agerange(2)& ~isnan(SGN(:,1))' & exclu & history ==0)
index1 = find(Age>Agerange(1) & Age<= Agerange(2)& ~isnan(SGN(:,1))' & exclu );
middleagemeanSGN = nanmean(SGN(index1,:)) ;
middleagesemSGN =  nanstd(SGN(index1,:))/sqrt(length(index1));
Agerange = [75,120];
% index1 = find(Age>Agerange(1) & Age<= Agerange(2)& ~isnan(SGN(:,1))' & exclu & history ==0)
index1 = find(Age>Agerange(1) & Age<= Agerange(2)& ~isnan(SGN(:,1))' & exclu );
OldmeanSGN = nanmean(SGN(index1,:));
OldsemSGN =  nanstd(SGN(index1,:))/sqrt(length(index1));

B_1 = [0,0,0]; % young 0-1
B_2 = [0,0,0]+0.45; % 1-50 yrs
B_3 = [0,0,0]+0.65; % 50-75 yrs
B_4 = [0,0,0]+0.9; % 75-100 yrs
freq = 165.4 * (10.^(2.1.*(100- [(EDGES(1:end-1)+EDGES(2:end))/2])/100) - 0.4)./1000;
IHCper10percent = 3.2*(89.4 + 0.182 * [(EDGES(1:end-1)+EDGES(2:end))/2] );
fig= figure('position',[1033 457 750 450]);sz=10; hold on 
ploterrorbar(freq,10*youngmeanSGN./IHCper10percent,10*youngsemSGN./IHCper10percent,B_1,sz)
ploterrorbar(freq,10*youngadultmeanSGN./IHCper10percent,10*youngadultsemSGN./IHCper10percent,B_2,sz)
ploterrorbar(freq,10*middleagemeanSGN./IHCper10percent,10*middleagesemSGN./IHCper10percent,B_3,sz)
ploterrorbar(freq,10*OldmeanSGN./IHCper10percent,10*OldsemSGN./IHCper10percent,B_4,sz)
legend({'Age < 10','Age 10-50','Age 50 - 75','Age >75'})
ylabel('SGN per IHC position')
xlabel('% from base')
% print(fig,'-dpdf')


%% Figure 7 plot the case in column XX (column 40 and column 41)
% step1; find the norm 
close all
cd(figurepath)
load('MLpaper2022data.mat')
m=0;bin = 5;
figure; hold on 
for columnN = 2:175
  if ~isempty(database{82,columnN}) & ~isempty(database{71,columnN}) & ~isempty(database{15,columnN})
    m=m+1;
    floorbacktoOC = database{82,columnN}.percentfrombasematchtov6data;
    EDGES = [0:bin:80,100];[NSGNeachbin4,~, ~] = histcounts(floorbacktoOC./10, EDGES);
    plot([2.5:bin:80,90],NSGNeachbin4)
    AllSGNs(m,:) = NSGNeachbin4;
    AllANF(m,:) = [database{15:19,columnN}];
  end
end
for i =1 : 17
normalby95percent(i) = prctile(AllSGNs(:,i),95);
end
plot([2.5:bin:80,90],normalby95percent)
for i =1 : 5
normalby95percentANF(i) = prctile(AllANF(:,i),98);
end
for columnN =41
    if ~isempty(database{82,columnN})
    SGCfreq = database{82,columnN}.percentfrombasematchtov6data;
    EDGES = [0:bin:80,100];[NSGNeachbin4,~, ~] = histcounts(SGCfreq./10, EDGES);
    fig= figure('position',[0 0 300 1200],'Name',['Tray',num2str(database{80,columnN})])
    
    subplot(20,1,[1:5]);hold on 
    frequencyplots([0.25,0.5,1,2,4,8],[database{25:30,columnN}],'o',[0 0.8 1],6.5)
    set(gca,'ydir','reverse');title(['Tray',num2str(database{80,columnN}),'  Age',num2str(database{5,columnN})])

    subplot(20,1,[6,7,8,9]);hold on
    freq = 165.4 * (10.^(2.1.*(100- [2.5:5:97.5])/100) - 0.4)./1000;
    frequencyplots(freq,100*database{33,columnN}(:,1),'o',[0 0 0.7],6.5)
    frequencyplots(freq,100*database{33,columnN}(:,5),'o',[0 0.4 1],6.5)
    title(['Noise = ',num2str(database{11,columnN}),'  TAI = ',num2str(database{12,columnN})])
    
    subplot(20,1,[10:14]);hold on
    x = 165.4 * (10.^(2.1.*(100- [2.5:bin:80,90])/100) - 0.4)./1000;
    frequencyplots( x,100*NSGNeachbin4./normalby95percent,'o',[0 0 0.7],6.5)
    frequencyplots( [7.5,1.8,0.7,0.3,0.17],100*[[database{15:19,columnN}]./normalby95percentANF],'o',[0 0.4 1],6.5)
    ylim([0,120]);title(['column',num2str(columnN)])
    
    subplot(20,1,[15:20]);hold on
    x = 165.4 * (10.^(2.1.*(100- [2.5:bin:80,90])/100) - 0.4)./1000;
    frequencyplots( x,NSGNeachbin4,'o',[0 0 0.7],6.5)
    ylim([0,250]);title(['column',num2str(columnN)])
%     print(gcf,'-dpdf')
end

end


%%  Part II - figure 8A & B - age related SGN, IHC, OHC, ANF loss 
exclusion = [11,20,21,60,61,62,63,64,65,66,67,68,69,71,72,73,74,75,...
    76,77,78,79,80,95,98,99,102,103,106,120,122,126,130,135,136,137,32,125,138]+1;
for i = 2:175
    if ~any(i == exclusion)
       exclu(i) = true;
    else
        exclu(i) = false;
    end
 
end

len = 20;row = 82;EDGES = [0:5:100];
for i = 2:175
    TAI(i) = database{12,i};
    if isnumeric(database{11,i})
        history(i) = database{11,i};
    else
        history(i) = NaN;
    end
    if isnumeric(database{5,i} )& ~isnan(database{5,i})
        Age(i) = database{5,i};
        WRS(i) = database{8,i};
    else
        Age(i) = NaN;
        WRS(i) = NaN;
    end
    if ~isempty(database{row,i})
        SGCfreq = database{row,i}.percentfrombasematchtov6data;
        [NSGNeachbin4,~, ~] = histcounts(SGCfreq./10, EDGES);
        SGN(i,1:length(NSGNeachbin4)) = NSGNeachbin4;
        else
        SGN(i,1:len) = NaN;
    end  
    
end
youngindex = 6% find(Age <6& ~isnan(SGN(:,1))' & exclu )
normalizedSGN = SGN%./(SGN(youngindex,:));
SGNcenteredatAudio(:,6) = nanmean(SGN(:,[4,5])')'
SGNcenteredatAudio(:,5) = nanmean(SGN(:,[7,8])')'
SGNcenteredatAudio(:,4) = nanmean(SGN(:,[10,11])')'
SGNcenteredatAudio(:,3) = nanmean(SGN(:,[12,13])')'
SGNcenteredatAudio(:,2) = nanmean(SGN(:,[15,16])')'
SGNcenteredatAudio(:,1) = nanmean(SGN(:,[17,18])')'
clear SGN ANF TAI IHC OHC
row = 82;len = 5
for i = 2:175
    if ~isempty(database{row,i})


        IHC(i,:) =  database{46,i}(:,1);
        OHC(i,:) =  database{46,i}(:,5);
        audio(i,:) = [database{25:30,i}];
        ANF(i,1) = database{55,i};
        ANF(i,2) =  nanmean([database{49,i},database{50,i}]);
        ANF(i,3) =  database{57,i};
        ANF(i,4) =  database{48,i};
        ANF(i,5) =  database{52,i}; 
        ANF(i,6) =  database{47,i};
        else

        OHC(i,1:6) = NaN;
        IHC(i,1:6) = NaN;   
        ANF(i,1:6) = NaN;
        audio(i,1:6) = NaN;

    end  
    
end
cd(figurepath)
freqran = [4:6] ; % [1:3] is 025kHz to 1kHz; [4:6] is 2kHz to 8kHz. 
meanSGN = mean(SGNcenteredatAudio(:,freqran)');
meanANF = mean(ANF(:,freqran)');
meanIHC = mean(IHC(:,freqran)');
meanOHC = mean(OHC(:,freqran)');
meannormalizeSGN = meanSGN./prctile(meanSGN,95);
% casetoplot = ~isnan(SGN(:,1) )' & exclu & history ==0;
casetoplot = ~isnan(SGNcenteredatAudio(:,1) )' & exclu;
B_1 = [0,0,0]; % young 0-1
B_2 = [0,0,0]+0.45 % 1-50 yrs
B_3 = [0,0,0]+0.65 % 50-75 yrs
B_4 = [0,0,0]+0.9 % 75-100 yrs

fig = figure; 
hold on 
plotscatterplot(Age(casetoplot),meanIHC(casetoplot),B_2,'k',20)
plotscatterplot(Age(casetoplot),meanOHC(casetoplot),B_4,'k',20)
plotscatterplot(Age(casetoplot),meanANF(casetoplot),'r','k',50)
plotscatterplot(Age(casetoplot),meannormalizeSGN(casetoplot),'b','k',50)

%% Figure 8 statistics: statistics in Steiger's Z test 
exclusion = [11,20,21,60,61,62,63,64,65,66,67,68,69,71,72,73,74,75,...
    76,77,78,79,80,95,98,99,102,103,106,120,122,126,130,135,136,137,32,125,138]+1;
for i = 2:175
    if ~any(i == exclusion)
       exclu(i) = true;
    else
        exclu(i) = false;
    end
 
end
len = 20;row = 82;EDGES = [0:50:100];
for i = 2:175
    TAI(i) = database{12,i};
    if isnumeric(database{11,i})
        history(i) = database{11,i};
    else
        history(i) = NaN;
    end
    if isnumeric(database{5,i} )& ~isnan(database{5,i})
        Age(i) = database{5,i};
        WRS(i) = database{8,i};
    else
        Age(i) = NaN;
        WRS(i) = NaN;
    end
    if ~isempty(database{row,i})
        SGCfreq = database{row,i}.percentfrombasematchtov6data;
        [NSGNeachbin4,~, ~] = histcounts(SGCfreq./10, EDGES);
        SGN(i,1:length(NSGNeachbin4)) = NSGNeachbin4;
        else
        SGN(i,1:len) = NaN;
    end  
    
end
clear SGN ANF TAI IHC OHC 
row = 82;len = 5
for i = 2:175
    if ~isempty(database{row,i})


        IHC(i,:) =  database{46,i}(:,1);
        OHC(i,:) =  database{46,i}(:,5);
        audio(i,:) = [database{25:30,i}];
        ANF(i,1) = database{55,i};
        ANF(i,2) =  nanmean([database{49,i},database{50,i}]);
        ANF(i,3) =  database{57,i};
        ANF(i,4) =  database{48,i};
        ANF(i,5) =  database{52,i}; 
        ANF(i,6) =  database{47,i};
        SGCfreq = database{row,i}.percentfrombasematchtov6data;
        totalSGN(i) = length(SGCfreq);
        [NSGNeachbin4,~, ~] = histcounts(SGCfreq./10, EDGES);
        halfSGC(i,:) =  NSGNeachbin4;
         
        else

        OHC(i,1:6) = NaN;
        IHC(i,1:6) = NaN;   
        ANF(i,1:6) = NaN;
        audio(i,1:6) = NaN;
        halfSGC(i,1:2) = NaN;

    end  
    
end

freqran = [1:3] ; % [1:3] is 025kHz to 1kHz; [4:6] is 2kHz to 8kHz. 
meanANFapex = nanmean(ANF(:,freqran)');
freqran = [4:6] ; % [1:3] is 025kHz to 1kHz; [4:6] is 2kHz to 8kHz. 
meanANFbase = nanmean(ANF(:,freqran)');
normalizedhalfSGC = halfSGC./max(halfSGC);
totalSGC = sum(halfSGC')./max(sum(halfSGC'));
% age related SGC loss in the entire cochlea
fitlm(Age(casetoplot),totalSGC(casetoplot))
% age related SGC loss in the basal half of cochlea
fitlm(Age(casetoplot),normalizedhalfSGC(casetoplot,1))
% age related SGC loss in the apical half of cochlea
fitlm(Age(casetoplot),normalizedhalfSGC(casetoplot,2))
% age related ANF loss in the apical half of cochlea
fitlm(Age(casetoplot),meanANFapex(casetoplot))
% age related ANF loss in the basal half of cochlea
fitlm(Age(casetoplot),meanANFbase(casetoplot))

% z test for SGN apex vs SGN base 
r1 = corrcoef(Age(casetoplot),halfSGC(casetoplot,1))
r2 = corrcoef(Age(casetoplot),halfSGC(casetoplot,2))
r3 = corrcoef(halfSGC(casetoplot,2),halfSGC(casetoplot,1))

% z test for the base SGN vs ANF 
% http://www.psychmike.com/dependent_correlations.php 
r1 = corrcoef(Age(casetoplot),meanANFbase(casetoplot))
r2 = corrcoef(Age(casetoplot),normalizedhalfSGC(casetoplot,1))
r3 = corrcoef(meanANFbase(casetoplot),normalizedhalfSGC(casetoplot,1))

% z test for the apex
% http://www.psychmike.com/dependent_correlations.php
r1 = corrcoef(Age(casetoplot),meanANFapex(casetoplot))
r2 = corrcoef(Age(casetoplot),normalizedhalfSGC(casetoplot,2))
r3 = corrcoef(meanANFapex(casetoplot),normalizedhalfSGC(casetoplot,2))

%% ML paper total count with age in Makary and Otte paper numbers used in method section 
ageotte = [5:10:85];
Ottetotal = [36918,30658,31914,27117,27872,25261,25270,22871,18626]
countMakarySegIV = [8408,7401,7281,7478,7085,6172,7116,6088,5810,5802];
countMakarySegI = [4479,4442,4154,3693,3237,3537,3279,3714,2686,2726];
countMakarySegIIlower = [6226,6016,5646,5267,4835,4951,5102,5220,4278,3828];
countMakarySegIIupper = [6620,5316,5585,5451,5125,5050,4892,4963,4363,4501];
countMakarySegIII = [7946,6944,7391,6929,6611,6096,6653,6215,5458,5587];
countMakarySegIV = [8408,7401,7281,7478,7085,6172,7116,6088,5810,5802];
IshiyamatotalSGN = [36480,48720,41280,46320,39120,36960]
Ishiyamaage = [16,35,38,57,77,80]
basal50Makary = countMakarySegI +countMakarySegIIlower+countMakarySegIIupper ;
Apical50Makary = countMakarySegIII + countMakarySegIV;
agemakary = [5:10:95];
% age related SGN count in Makary paper
makarytotal = basal50Makary+Apical50Makary;
fitlm(agemakary,makarytotal./max(makarytotal))
% age related SGN count in Otte paper 
fitlm(ageotte,Ottetotal./max(Ottetotal))


