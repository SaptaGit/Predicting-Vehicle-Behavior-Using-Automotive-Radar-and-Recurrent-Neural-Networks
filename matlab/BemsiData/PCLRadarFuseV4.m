clc
clear all
close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% Radar and lidar data extraction script %%%%%%%%%%%%%%%%%%
% Author: Saptarshi Mukherjee
% This script will generate the radar signal at an specific angle and
% extract the lidar points associated to each radar signal point
% The extracted data formats are explained below

%%%%%%%%%%%%%% Input %%%%%%%%%%%%%%%%
% Extracted Lidar, radar and Image file

%%%%%%%%%%%%%% Output %%%%%%%%%%%%%%%%
% 'lidarDict.mat' -> A Map object with Keys 1:1152, where each value holds
% the raw lidar points associated to that specific radar image cell. So
% bacisally lidar point values at '465' would be the lidar points against
% belongs to radar signal cell 465
% 'radarHitIndex.mat'-> This is a list of length 1152 where each cell has
% values either 0 or 1, where 1 indicated there are one or more lidar point
% exists.
% 'radarSignal.mat' -> Raw radar signal along the selected azimuth
% 'rawLidarXYZ.mat' -> Raw lidar points of the whole scene in case needed
% 'sceneImage.png' -> RGB image of the scene for refernce
% 'radarImage.png' -> Raw radar image of the scene

% 'frontLidarDict.mat' -> same as lidarDict.mat except in this dict index 1
% means closes too the car and index 576 means farthest from the car in
% forward lookiing direction
% 'frontradarHitIndex.mat'-> same as radarHitIndex.mat excpet only the 
% front of the car.
% 'frontRadarSignal.mat' -> same as radarSignal.mat except only the front
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%% Specify if you want to pick 'Row' or 'Column' %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rowStr = 'Row';
colStr = 'Column';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Chnage to rowStr for Row selection and colStr for coulmn selection %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rowOrCol = rowStr;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Specify the index from Radar Image Dimension POV %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rowOrColNum = 540;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Radar image dimension from the sensor
radarImageDim = 1152;
radarImageCentre = radarImageDim/2;

% meter per pixel in Radar
cellResolution = 0.173611; 

% Folder path to dump all the values as mat file
%dataSaveFolder = '/home/saptarshi/PythonCode/Junction/matlab/BemsiData/Output/';
dataSaveFolder = './Output/';

% load all relevent data
inputFilePath = './Input/';
radarImage = imread(strcat(inputFilePath,'NavtechImage.png'));
RGBImage = imread(strcat(inputFilePath,'RGBImage.png'));
veloCSV = load(strcat(inputFilePath,'Lidar.csv'));

% Extract velodyne XYZ + channel + intensity
VeloXYZ = [veloCSV(:,1),veloCSV(:,2),veloCSV(:,3)];
VeloIntensity = veloCSV(:,4);
VeloChannel = veloCSV(:,5);
maskedXYZ = VeloXYZ;

% Extract the ground plane from the velo plane
veloGround = [veloCSV(:,1), veloCSV(:,2)];
radarGround = veloGround;
radarGroundMasked = radarGround ;

% Create a dict with keys equal to (1-1152) all possible converted Y values
lidarDict = containers.Map();
for keyIndex = 1:radarImageDim
    lidarDict(num2str(keyIndex)) = [];
end

% Convert the velo points to Radar Co-ordinate
for idx = 1 : length(veloGround)
    % Convert each lidar point to Radar Image dimension and scale
    radarGround(idx,1) = radarImageCentre + int32(veloGround(idx,1)/cellResolution);
    radarGround(idx,2) = radarImageCentre - int32(veloGround(idx,2)/cellResolution);
    
    %  If the converted value's X val is 576 means it is on the red line
    % The extract the coresponding converted Y val and append all the raw
    % lidar points to the lidarDict against the Y val key
    convertedXVal = radarGround(idx,1);
    convertedYVal = radarGround(idx,2);
    rawLidarX = VeloXYZ(idx,1);
    rawLidarY = VeloXYZ(idx,2);
    rawLidarZ = VeloXYZ(idx,3);
    channelVal = VeloChannel(idx);
    intensityVal = VeloIntensity(idx);
    if strcmp(rowOrCol,colStr)
        if (convertedXVal == rowOrColNum)
            lidarDict(num2str(convertedYVal)) = [lidarDict(num2str(convertedYVal));[rawLidarX,rawLidarY,rawLidarZ,channelVal,intensityVal]];
        end
    elseif strcmp(rowOrCol,rowStr)      
        if (convertedYVal == rowOrColNum)
            lidarDict(num2str(convertedXVal)) = [lidarDict(num2str(convertedYVal));[rawLidarX,rawLidarY,rawLidarZ,channelVal,intensityVal]];
        end
    else
        disp('Unknown Row or Column Selection!!!')
        doc return
    end    
end

% Create the binary image using lidar points converted to radar image resolution
lidarBinaryImage = zeros(radarImageDim,radarImageDim,3);

% Create a binary image from lidar points same dimension as of radar
% Convert the velo points to Radar Co-ordinate
lidarBinary = veloGround;
for idx = 1 : length(veloGround)
    % Convert each lidar point to Radar Image dimension and scale
    lidarBinary(idx,1) = radarImageCentre + int32(veloGround(idx,1)/cellResolution);
    lidarBinary(idx,2) = radarImageCentre - int32(veloGround(idx,2)/cellResolution);
    lidarBinaryImage(lidarBinary(idx,1),lidarBinary(idx,2),1) = 255;
    lidarBinaryImage(lidarBinary(idx,1),lidarBinary(idx,2),2) = 0;
    lidarBinaryImage(lidarBinary(idx,1),lidarBinary(idx,2),3) = 0;
end


% Quick visualization move to the end
figure(1)
orientedLidarBinaryImage = flip(rot90(rot90(rot90(lidarBinaryImage))),2);
fusedImage = imfuse(radarImage,orientedLidarBinaryImage);
imshow(fusedImage)

% Get the Radar Signal
if strcmp(rowOrCol,colStr)
    radarSignal = radarImage(:,rowOrColNum);
elseif strcmp(rowOrCol,rowStr)  
    radarSignal = radarImage(rowOrColNum,:);
else
    disp('Unknown Row or Column Selection!!!')
    doc return
end

% Get the corresponding Lidar hit yes or no
hitIndex = zeros(1,radarImageDim);
for radarIndex = 1:radarImageDim
    currentPoints = lidarDict(num2str(radarIndex));
    currentPointSize = size(currentPoints,1);
    if(currentPointSize ~= 0)
        hitIndex(radarIndex) = 1;
    end 
end


% Do the Velo points selection based on the line
% This is veloo points selection but with a different approach
% Its kept at the moment to make sure the other approach is working as
% expetced
for idx = 1 : length(maskedXYZ)
    currentXVal = maskedXYZ(idx,1);
    currentYVal = maskedXYZ(idx,2);
    currentPoint = [currentXVal,currentYVal];
    
    % Calculate the real world co rodinate to filter the lidar points
    realWorldPoint = (rowOrColNum-radarImageCentre)*cellResolution ;
    
    if strcmp(rowOrCol,colStr)
        if currentXVal > (realWorldPoint-cellResolution/2) && currentXVal < (realWorldPoint+cellResolution/2)
            maskedXYZ(idx,1) = 0;
            maskedXYZ(idx,2) = 1;
            maskedXYZ(idx,3) = 0;
        else
            maskedXYZ(idx,1) = 1;
            maskedXYZ(idx,2) = 0;
            maskedXYZ(idx,3) = 0;
        end
    elseif strcmp(rowOrCol,rowStr)       
        if currentYVal > (realWorldPoint-cellResolution/2) && currentYVal < (realWorldPoint+cellResolution/2)
            maskedXYZ(idx,1) = 0;
            maskedXYZ(idx,2) = 1;
            maskedXYZ(idx,3) = 0;
        else
            maskedXYZ(idx,1) = 1;
            maskedXYZ(idx,2) = 0;
            maskedXYZ(idx,3) = 0;
        end
    else
        disp('Unknown Row or Column Selection!!!')
        doc return
    end
end


% Do sanity check operations for lidar before plot
% For sanity check if we do 3D plot of all the raw lidar points it should
% be the same as the masked 
cellWiseLidarPoints = [];
for collectionIndex = 1:radarImageDim
    currentPoints = lidarDict(num2str(collectionIndex));
    for addIndex = 1:size(currentPoints,1)
        cellWiseLidarPoints = [cellWiseLidarPoints; currentPoints(addIndex,1:3)];
    end
end


% Flip the radar and hitIndex file to get only the front of the car
flippedRadar = flip(radarSignal);
frontRadarSignal = flippedRadar(577:1152);

flippedHitIndex = flip(hitIndex);
frontHitIndex = flippedHitIndex(577:1152);

% Create the new lidarDict with index from 576 to 1 as 1 to 576 
frontLidarDict = containers.Map();
for frontDictIndex = 1:576
    frontLidarDict(num2str(frontDictIndex)) = lidarDict(num2str(577-frontDictIndex));
end


% Do all the plots
figure(1);
title('Radar Image, Projected Lidar and Selcted Azimuth')
% Plot the radar image, projected ground truth and selcted signal
% c = [1 0 0];
% sz = 2;
imshow(radarImage*1.2)
hold on
scatter(radarGround(:,1),radarGround(:,2), 3, [0.8 0.8 0]);
% If single col draw line else draw box
if strcmp(rowOrCol,colStr)
    line([rowOrColNum, rowOrColNum],[0, radarImageDim], 'color', 'red');
elseif strcmp(rowOrCol,rowStr)
    line([0, radarImageDim],[rowOrColNum, rowOrColNum], 'color', 'red');
else
    disp('Unknown Row or Column Selection!!!')
    doc return
end

%
figure(2);
title('Radar Image, Projected Lidar and Selcted Azimuth')
% Plot the radar image, projected ground truth and selcted signal
c = [1 0 0];
sz = 2;
imshow(radarImage*1.2)
hold on
% If single col draw line else draw box
if strcmp(rowOrCol,colStr)
    line([rowOrColNum, rowOrColNum],[0, radarImageDim], 'color', 'red');
elseif strcmp(rowOrCol,rowStr)
    line([0, radarImageDim],[rowOrColNum, rowOrColNum], 'color', 'red');
else
    disp('Unknown Row or Column Selection!!!')
    doc return
end

% plot the Radar signal
maxRadarSignalStrength = double(max(radarSignal))/2;
figure(3);
plot(radarSignal,'LineWidth',2)
hold on
plot(hitIndex*maxRadarSignalStrength,'LineWidth',2)
xlabel('Range (meter)','FontSize', 20)
ylabel('Intensity/Lidar Hit','FontSize', 20)
xticks([1,288,576,864,1152])
xticklabels([+100,+50,0,-50,-100])
ax = gca;
ax.FontSize = 16; 
title('Radar Signal and coresponding lidar hit','FontSize', 25)

% plot the Color Masked PCL
figure(4);
pcshow(VeloXYZ,maskedXYZ)
pause(5)
hold on
pcshow(cellWiseLidarPoints,[1,1,1])

% plot the Front Radar signal and coresponding hitIndex
maxRadarSignalStrength = double(max(radarSignal))/2;
figure(5);
plot(frontRadarSignal,'LineWidth',2)
hold on
plot(frontHitIndex*maxRadarSignalStrength,'LineWidth',2)
xlabel('Range (meter)','FontSize', 20)
ylabel('Intensity/Lidar Hit','FontSize', 20)
xticks([1,288,576])
xticklabels([0,+50,100])
ax = gca;
ax.FontSize = 16; 
title('Front Radar Signal and coresponding lidar hit','FontSize', 25)



% Save all the data
lidarDictFileName = strcat(dataSaveFolder, 'lidarDict.mat');
save(lidarDictFileName,'lidarDict');
radarSignalFileName = strcat(dataSaveFolder, 'radarSignal.mat');
save(radarSignalFileName,'radarSignal');
radarHitIndexFileName = strcat(dataSaveFolder, 'radarHitIndex.mat');
save(radarHitIndexFileName,'hitIndex');
rawLidarPointsFileName = strcat(dataSaveFolder, 'rawLidarXYZ.mat');
save(rawLidarPointsFileName,'VeloXYZ');
frontLidarDictFileName = strcat(dataSaveFolder, 'frontLidarDict.mat');
save(frontLidarDictFileName,'frontLidarDict');
frontRadarSignalFileName = strcat(dataSaveFolder, 'frontRadarSignal.mat');
save(frontRadarSignalFileName,'frontRadarSignal');
frontRadarHitIndexFileName = strcat(dataSaveFolder, 'frontRadarHitIndex.mat');
save(frontRadarHitIndexFileName,'frontHitIndex');

% Save all scene infos (images)
rgbImageFileName = strcat(dataSaveFolder, 'sceneImage.png');
imwrite(RGBImage,rgbImageFileName);
rawRadarImageFileName = strcat(dataSaveFolder, 'radarImage.png');
imwrite(radarImage,rawRadarImageFileName);



