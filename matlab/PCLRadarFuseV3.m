clc
clear all
close all

radarImageDim = 1152;
radarImageCentre = radarImageDim/2;

% meter per pixel in Radar
cellResolution = 0.173611; 
signalCount = 1;


radarImage = imread('/home/saptarshi/bags/loam/SmallLoam/Navtech_Cartesian/000006.png');
RGBImage = imread('/home/saptarshi/bags/loam/SmallLoam/zed_left/000004.png');
veloCSV = load('/home/saptarshi/bags/loam/SmallLoam/velo_lidar/000036.csv');
VeloXYZ = [veloCSV(:,1),veloCSV(:,2),veloCSV(:,3)];
maskedXYZ = VeloXYZ;

% Extract the ground plane from the velo plane
veloGround = [veloCSV(:,1), veloCSV(:,2)];
radarGround = veloGround;
radarGroundMasked = radarGround ;

% create a dict with Y 1-1152 all possible Y values and a list for each Y
% key
lidarDict = containers.Map();
for keyIndex = 1:radarImageDim
    lidarDict(num2str(keyIndex)) = [];
end

% Convert the velo points to Radar Co-ordinate
for idx = 1 : length(veloGround)
    radarGround(idx,1) = radarImageCentre + int32(veloGround(idx,1)/cellResolution);
    radarGround(idx,2) = radarImageCentre - int32(veloGround(idx,2)/cellResolution);
    
    % new check....
    radarGroundMasked(idx,1) = radarImageCentre + int32(veloGround(idx,1)/cellResolution);
    radarGroundMasked(idx,2) = radarImageCentre - int32(veloGround(idx,2)/cellResolution);
    
    %  if the values are radar ground 576 
    % check radar Y and add all lidar (raw x,y) that to dict with Y as key
    % add the raw x,y in the dict 
    convertedYVal = radarGround(idx,2);
    convertedXVal = radarGround(idx,1);
    rawLidarX = VeloXYZ(idx,1);
    rawLidarY = VeloXYZ(idx,2);
    rawLidarZ = VeloXYZ(idx,3);
    if (convertedXVal == 576)
        lidarDict(num2str(convertedYVal)) = [lidarDict(num2str(convertedYVal));[rawLidarX,rawLidarY,rawLidarZ]];
    end

    if (radarGround(idx,1) ~= 576)
        radarGroundMasked(idx,1) = 0;
        radarGroundMasked(idx,2) = 0;
    end
    
end

% once done sort the dict keys which converted Y 
% for each Y key in dict all the raw lidar x,y s are the point in that cell


% now if we do 3d plot all all raw xy s in the dict we should have same as
% the masked lidar to check 

cellWiseLidarPoints = [];
for collectionIndex = 1:radarImageDim
    currentPoints = lidarDict(num2str(collectionIndex));
    for addIndex = 1:size(currentPoints,1)
        cellWiseLidarPoints = [cellWiseLidarPoints; currentPoints(addIndex,:)];
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



% Do the average thing based on the col count




% Do the Velo points selection based on the line
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



% Do all the plots

figure(1);
title('Radar Image, Projected Lidar and Selcted Azimuth')
% Plot the radar image, projected ground truth and selcted signal
c = [1 0 0];
sz = 2;
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
scatter(radarGroundMasked(:,1),radarGroundMasked(:,2), 3, [0.8 0.8 0]);
% If single col draw line else draw box
if signalCount == 1
    line([radarImageCentre, radarImageCentre],[0, radarImageDim], 'color', 'red');
else
    line([radarImageCentre-signalCount,0],[radarImageCentre-signalCount,radarImageDim], 'color', 'red');
    line([radarImageCentre+signalCount,0],[radarImageCentre+signalCount,radarImageDim], 'color', 'red');
end

% plot the Radar signal
figure(3);
plot(radarSignal)
xlabel('Range')
ylabel('Intensity')
title('Radar Signal')

% plot the Color Masked PCL
figure(4);
pcshow(VeloXYZ,maskedXYZ)
hold on
pcshow(cellWiseLidarPoints,[0,0,1])



