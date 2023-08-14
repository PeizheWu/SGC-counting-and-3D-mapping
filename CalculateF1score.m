Groundtruthpath = '\\apollo\research\ENT\Liberman\Peizhe Wu\06 Publications\Work in progress\8 SGC technique paper\01 Code and data for figures v12\ML models and training data\Calculating the F1 score of the model\Data to compare F1 score to ground truth\Groundtruth'
GT = dir(Groundtruthpath);
MLresultPath = '\\apollo\research\ENT\Liberman\Peizhe Wu\06 Publications\Work in progress\8 SGC technique paper\01 Code and data for figures v12\ML models and training data\Calculating the F1 score of the model\Data to compare F1 score to ground truth\ML result'
MLresult = dir(MLresultPath)
clear similarity
% calculate Dice coefficient 
for caseN =3:13 %3:29
    cd(Groundtruthpath)
    GTimag = imbinarize(imread(GT(caseN).name));
    cd(MLresultPath)
    MLimag =imbinarize(imread(MLresult(caseN).name));
    MLimag = MLimag(:,9:3608);
    if length(unique(GTimag))<2
        similarity(caseN-2) = NaN;
    else
        similarity(caseN-2)  = dice(GTimag,MLimag);
    end
end
nanmean(similarity)
% calculate F1 score 
for caseN =3:13
    cd(Groundtruthpath)
    GTimag = imbinarize(imread(GT(caseN).name));
    CCGT = bwconncomp(GTimag,8)
    
    cd(MLresultPath)
    MLimag =imbinarize(imread(MLresult(caseN).name));
    MLimag = MLimag(:,9:3608);
    CCML = bwconncomp(MLimag,8)
    
    
     if length(unique(GTimag))<2
         F1score(caseN-2) = NaN;
     else
         
        clear specificity sensitivity
        for i = 1 : length( CCML.PixelIdxList) 
            MLlist = CCML.PixelIdxList{i};
            clear overlap
            for k = 1:length(MLlist)
                overlap(k) = GTimag(MLlist(k));
            end
            Specificity(i) = logical(sum(overlap)>0);
        end

        for i = 1 : length( CCGT.PixelIdxList) 
            GTlist = CCGT.PixelIdxList{i};
            clear overlap
            for k = 1:length(GTlist)
                overlap(k) = MLimag(GTlist(k));
            end
            sensitivity(i) = logical(sum(overlap)>0);
        end

        TPN = sum(Specificity)
        FPN = length(Specificity)-sum(Specificity)
        FN = length(sensitivity)-sum(sensitivity)

        Presicion = TPN ./(TPN + FPN);
        Recall = TPN./(TPN + FN);
        F1score(caseN-2) = 2*Presicion*Recall./(Presicion + Recall);
     end
end
nanmean(F1score)
nanstd(F1score)./sqrt(11)

%% the following part is to calculate F1 score between Neurolucida and ML 
% Copy the folder 'Correlate Neurolucida with ML result' to desktop to run
% the following script 

% load data and calculate F1 score for each spiral ganglion segement. 
basedir = 'C:\Users\wupeizh\Desktop\Correlate Neurolucida with ML result\Results Zurek R'
MLresultfolder = [basedir,'\','EveryMLfoundSGN'];
cd(MLresultfolder)
MLresultsdir = dir(MLresultfolder);
Neuroresultfolder = [basedir,'\','EveryNeurolucidafoundSGN']
cd(Neuroresultfolder)
Neurolucidaresultsdir  = dir(Neuroresultfolder);

for segment  = 3:34
    ML = readtable([MLresultsdir(segment).folder,'\',MLresultsdir(segment).name]);
    Neuro = readtable([Neurolucidaresultsdir(segment).folder,'\',Neurolucidaresultsdir(segment).name]);
    TP = length(find(ML.distanceSGN_foreveryMLfoundSGN <30)); 
    FP = length(find(ML.distanceSGN_foreveryMLfoundSGN>30));
    FN = length(find(Neuro.distanceSGN_foreveryNeurolucidaSGN>30));
    Presicion = TP./(TP+FP);
    Recall = TP./(TP+FN);
    F1score(segment-2) = 2*Presicion * Recall/(Presicion + Recall);    
end
    
    

% The ML result don't over lap with Neurolucida 100% because of the
% stage drifting. Based on the histogram of distance to the closes
% location, the datapoint where the distance to the nearest point on
% another method <30 is considered a true positive. 

figure; hold on 
histogram(MLresult.distanceSGN_foreveryMLfoundSGN)
histogram(NeuroResult.distanceSGN_foreveryNeurolucidaSGN)

index = find(MLresult.distanceSGN_foreveryMLfoundSGN >30)
index2 = find(NeuroResult.distanceSGN_foreveryNeurolucidaSGN >30)

figure; hold on 
scatter(MLresult.Xcor,MLresult.Ycor,'ro')
scatter(NeuroResult.Xcor,NeuroResult.Ycor,'bo')
scatter(MLresult.Xcor(index),MLresult.Ycor(index),'r*')
scatter(NeuroResult.Xcor(index2),NeuroResult.Ycor(index2),'b*')




