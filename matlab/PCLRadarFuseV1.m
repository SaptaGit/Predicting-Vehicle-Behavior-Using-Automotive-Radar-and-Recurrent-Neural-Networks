clc
clear all
close all

radarImageDim = 1152;
radarImageCentre = radarImageDim/2;


radarImage = imread('/home/saptarshi/registration/SensorFusionAnnotationTool/sequences/gorgie_2_small/Navtech_Cartesian/000213.png');
veloCSV = load('/home/saptarshi/registration/SensorFusionAnnotationTool/sequences/gorgie_2_small/velo_lidar/000557.csv');
VeloXYZ = [veloCSV(:,1),veloCSV(:,2),veloCSV(:,3)];
maskedXYZ = VeloXYZ;

cellResolution = 0.173611;

x1 = 457;
y1 = 14;

x2 = 646;
y2 = 9;

x1Shift = radarImageCentre - x1;
y1Shift = radarImageCentre - y1;

x2Shift = radarImageCentre - x2;
y2Shift = radarImageCentre - y2;

worldX1 = x1Shift*cellResolution;
worldY1 = y1Shift*cellResolution;

worldX2 = x2Shift*cellResolution;
worldY2 = y2Shift*cellResolution;

% corner1 = [10 -0.5];
% corner2 = [5 7.5];
% corner3 = [0 0];

corner1 = [worldX1 worldY1];
corner2 = [worldX2 worldY2];
corner3 = [0 0];

for idx = 1 : length(maskedXYZ)
    currentXVal = maskedXYZ(idx,1);
    currentYVal = maskedXYZ(idx,2);
    currentPoint = [currentXVal,currentYVal];
    
    retVal = f_check_inside_triangle(corner1,corner2,corner3, currentPoint);
    
    if(retVal == 0)
        maskedXYZ(idx,1) = 1;
        maskedXYZ(idx,2) = 0;
        maskedXYZ(idx,3) = 0;
    else
        maskedXYZ(idx,1) = 0;
        maskedXYZ(idx,2) = 0;
        maskedXYZ(idx,3) = 1;
    end
end


figure(1);
imshow(radarImage)
hold on
line([radarImageDim/2 x1],[radarImageDim/2 y1],'LineWidth',2)
hold on
line([radarImageDim/2 x2],[radarImageDim/2 y2],'LineWidth',2)
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
