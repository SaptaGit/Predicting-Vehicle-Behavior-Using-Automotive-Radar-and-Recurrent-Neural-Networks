%tSNE plot for NGSIM classification results
clear all 
clc


statesFolderPath = '/media/saptarshi/Storage/system_backup/PythonCode/Junction/EncodedStates/SampleStatesAndClassesJunctionV1Run5/ValidationData';

 % Read the list of files from the folder
 fileList=dir([statesFolderPath '/*.txt']);
 lstmCellStateCount = 256;
 
 % Check the number of files and then create arrays of same length to sore
 % the values
 itemLength = length(fileList);
 stateH1Array = zeros(itemLength,lstmCellStateCount);
 stateC1Array = zeros(itemLength,lstmCellStateCount);
 stateH2Array = zeros(itemLength,lstmCellStateCount);
 stateC2Array = zeros(itemLength,lstmCellStateCount);
 trueClasses = zeros(itemLength,1);
 
 
 % Loop through the files and read individual files 
 for k=1:length(fileList)
     % Create the file path
    filename=[statesFolderPath '/' fileList(k).name];
    
    % Read the content of the currennt file and store into corresponding
    % arrays
    fileText = fileread(filename);
    
    % Split the file text at new line to ake 5 speprate items one class and
    % four states
    splittedLines = splitlines(fileText);
    
    % The firs items is the class infor
    currentClass = str2double(splittedLines(1));
    trueClasses(k) = currentClass;
    
    % The second item is the state C1
    stateC1StrsList = split(splittedLines(2));
    stateC1DoubleList = str2double(stateC1StrsList);
    stateC1Array(k,:) = stateC1DoubleList;
    
    % Next do your operation and finding
 end

% Do the tSNE plot
stateC1tSNEReduced = tsne(stateC1Array);

% Plo the reduced dimension against the true classes
gscatter(stateC1tSNEReduced(:,1),stateC1tSNEReduced(:,2),trueClasses)







