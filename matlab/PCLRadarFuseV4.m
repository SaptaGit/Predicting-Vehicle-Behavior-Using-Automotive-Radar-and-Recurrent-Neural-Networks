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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Radar image dimension from the sensor
radarImageDim = 1152;
radarImageCentre = radarImageDim/2;

% meter per pixel in Radar
cellResolution = 0.173611; 
signalCount = 1;

% Folder path to dump all the values as mat file
dataSaveFolder = '/home/saptarshi/PythonCode/Junction/matlab/BemsiData/Output/';

% load all relevent data
inputFilePath = '/home/saptarshi/PythonCode/Junction/matlab/BemsiData/Input/';
radarImage = imread(strcat(inputFilePath,'NavtechImage.png'));
RGBImage = imread(strcat(inputFilePath,'RGBImage.png'));
veloCSV = load(strcat(inputFilePath,'Lidar.csv'));

% Extract velodyne XYZ poses and Ignore Channel and intensity
VeloXYZ = [veloCSV(:,1),veloCSV(:,2),veloCSV(:,3)];
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
    convertedYVal = radarGround(idx,2);
    convertedXVal = radarGround(idx,1);
    rawLidarX = VeloXYZ(idx,1);
    rawLidarY = VeloXYZ(idx,2);
    rawLidarZ = VeloXYZ(idx,3);
    if (convertedXVal == 576)
        lidarDict(num2str(convertedYVal)) = [lidarDict(num2str(convertedYVal));[rawLidarX,rawLidarY,rawLidarZ]];
    end
    
end

% Get the Radar Signal
colCount = signalCount-1;
radarSignal = radarImage(:,radarImageCentre-colCount:radarImageCentre+colCount);

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
    
    if currentXVal > -cellResolution/2 && currentXVal < cellResolution/2
        maskedXYZ(idx,1) = 1;
        maskedXYZ(idx,2) = 0;
        maskedXYZ(idx,3) = 0;
    else
        maskedXYZ(idx,1) = 0;
        maskedXYZ(idx,2) = 1;
        maskedXYZ(idx,3) = 0;
        
    end
end


% Do sanity check operations for lidar before plot
% For sanity check if we do 3D plot of all the raw lidar points it should
% be the same as the masked 
cellWiseLidarPoints = [];
for collectionIndex = 1:radarImageDim
    currentPoints = lidarDict(num2str(collectionIndex));
    for addIndex = 1:size(currentPoints,1)
        cellWiseLidarPoints = [cellWiseLidarPoints; currentPoints(addIndex,:)];
    end
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
if signalCount == 1
    line([radarImageCentre, radarImageCentre],[0, radarImageDim], 'color', 'red');
else
    line([radarImageCentre-signalCount,0],[radarImageCentre-signalCount,radarImageDim], 'color', 'red');
    line([radarImageCentre+signalCount,0],[radarImageCentre+signalCount,radarImageDim], 'color', 'red');
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
if signalCount == 1
    line([radarImageCentre, radarImageCentre],[0, radarImageDim], 'color', 'red');
else
    line([radarImageCentre-signalCount,0],[radarImageCentre-signalCount,radarImageDim], 'color', 'red');
    line([radarImageCentre+signalCount,0],[radarImageCentre+signalCount,radarImageDim], 'color', 'red');
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
xticklabels([-100,-50,0,50,100])
ax = gca;
ax.FontSize = 16; 
title('Radar Signal and coresponding lidar hit','FontSize', 25)

% plot the Color Masked PCL
figure(4);
pcshow(VeloXYZ,maskedXYZ)
pause(5)
hold on
pcshow(cellWiseLidarPoints,[0,0,1])


% Save all the data
lidarDictFileName = strcat(dataSaveFolder, 'lidarDict.mat');
save(lidarDictFileName,'lidarDict');
radarSignalFileName = strcat(dataSaveFolder, 'radarSignal.mat');
save(radarSignalFileName,'radarSignal');
radarHitIndexFileName = strcat(dataSaveFolder, 'radarHitIndex.mat');
save(radarHitIndexFileName,'hitIndex');
rawLidarPointsFileName = strcat(dataSaveFolder, 'rawLidarXYZ.mat');
save(rawLidarPointsFileName,'VeloXYZ');

% Save all scene infos (images)
rgbImageFileName = strcat(dataSaveFolder, 'sceneImage.png');
imwrite(RGBImage,rgbImageFileName);
rawRadarImageFileName = strcat(dataSaveFolder, 'radarImage.png');
imwrite(radarImage,rawRadarImageFileName);



