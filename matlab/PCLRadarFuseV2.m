clc
clear all
close all

radarImageDim = 1152;
radarImageCentre = radarImageDim/2;
cellResolution = 0.173611;

radarImage = imread('/home/saptarshi/bags/loam/SmallLoam/Navtech_Cartesian/000006.png');
veloCSV = load('/home/saptarshi/bags/loam/SmallLoam/velo_lidar/000036.csv');
VeloXYZ = [veloCSV(:,1),veloCSV(:,2),veloCSV(:,3)];
maskedXYZ = VeloXYZ;

tlX = 480;
tlY = 460;

trX = 620;
trY = 460;

brX = 620;
brY = 520;

blX = 480;
blY = 520;

tlXShift = radarImageCentre - tlX;
tlYShift = radarImageCentre - tlY;

trXShift = radarImageCentre - trX;
trYShift = radarImageCentre - trY;

brXShift = radarImageCentre - brX;
brYShift = radarImageCentre - brY;

blXShift = radarImageCentre - blX;
blYShift = radarImageCentre - blY;


tlXWorld = tlXShift*cellResolution;
tlYWorld = tlYShift*cellResolution;

trXWorld = trXShift*cellResolution;
trYWorld = trYShift*cellResolution;

brXWorld = brXShift*cellResolution;
brYWorld = brYShift*cellResolution;

blXWorld = blXShift*cellResolution;
blYWorld = blYShift*cellResolution;

xCorners = [tlXWorld trXWorld brXWorld blXWorld];
% xCorners = xCorners - 10; % for the co-ordinate transform
yCorners = [tlYWorld trYWorld brYWorld blYWorld];

% corner1 = [worldX1 worldY1];
% corner2 = [worldX2 worldY2];
% corner3 = [0 0];

for idx = 1 : length(maskedXYZ)
    currentXVal = maskedXYZ(idx,1);
    currentYVal = maskedXYZ(idx,2);
    currentPoint = [currentXVal,currentYVal];
    
    % retVal = f_check_inside_triangle(corner1,corner2,corner3, currentPoint);
    retVal = inpolygon(currentXVal,currentYVal,xCorners,yCorners);
    
    if(retVal == 0)
        maskedXYZ(idx,1) = 1;
        maskedXYZ(idx,2) = 0;
        maskedXYZ(idx,3) = 0;
    else
        maskedXYZ(idx,1) = 0;
        maskedXYZ(idx,2) = 1;
        maskedXYZ(idx,3) = 0;
    end
end



% tlX = 480;
% tlY = 460;
% 
% trX = 620;
% trY = 460;
% 
% brX = 620;
% brY = 520;
% 
% blX = 480;
% blY = 520;


figure(1);
imshow(radarImage)
hold on
line([tlX trX],[tlY trY],'LineWidth',2)
hold on
line([trX brX],[trY brY],'LineWidth',2)
hold on
line([brX blX],[brY blY],'LineWidth',2)
hold on
line([blX tlX],[blY tlY],'LineWidth',2)
figure(2);
pcshow(VeloXYZ, maskedXYZ)
% figure(3);
% pcshow(maskedXYZ)





% y_shift = -0.25;
% x_shift = 0.6;
% xyzi = zeros(size(radarImage,1)*size(radarImage,2),4);
% cnt  =1;
% for i=1:size(radarImage,1)
%     for j=1:size(radarImage,2)
%         if (radarImage(i,j) > 0)
%         xyzi(cnt,:) = [(j - 576)*cell_size+x_shift, (-i + 576)*cell_size + y_shift, -2.0, double(radarImage(i,j))];
%         cnt = cnt + 1;
%         end
%     end
% end
% 
% 
% xyzNew = veloCSV;
% xyzNew((xyzNew(:,3)<-1.5),:)=[];
% figure(2)
% plot(xyzi(:,1),xyzi(:,2),'r*'); hold on;
% plot(xyzNew(:,1),xyzNew(:,2),'*')
