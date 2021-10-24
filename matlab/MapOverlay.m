% Check map and radar image overlay

mapImage = imread('/home/saptarshi/PythonCode/Junction/SightHillMap.png');
resizedMap = imresize(mapImage, [1152, 1152]);
radarImage1 = imread('/home/saptarshi/Dropbox (Heriot-Watt University Team)/RES_EPS_PathCad/datasets/pathcad_van_dataset/SighthillJunc/SighthillJunc_0/Navtech_Cartesian/000001.png');
radarImage2 = imread('/home/saptarshi/Dropbox (Heriot-Watt University Team)/RES_EPS_PathCad/datasets/pathcad_van_dataset/SighthillJunc/SighthillJunc_0/Navtech_Cartesian/000002.png');
radarImage3 = imread('/home/saptarshi/Dropbox (Heriot-Watt University Team)/RES_EPS_PathCad/datasets/pathcad_van_dataset/SighthillJunc/SighthillJunc_0/Navtech_Cartesian/000003.png');
radarImage4 = imread('/home/saptarshi/Dropbox (Heriot-Watt University Team)/RES_EPS_PathCad/datasets/pathcad_van_dataset/SighthillJunc/SighthillJunc_0/Navtech_Cartesian/000004.png');
radarImage5 = imread('/home/saptarshi/Dropbox (Heriot-Watt University Team)/RES_EPS_PathCad/datasets/pathcad_van_dataset/SighthillJunc/SighthillJunc_0/Navtech_Cartesian/000005.png');
radarImage6 = imread('/home/saptarshi/Dropbox (Heriot-Watt University Team)/RES_EPS_PathCad/datasets/pathcad_van_dataset/SighthillJunc/SighthillJunc_0/Navtech_Cartesian/000006.png');
radarImage7 = imread('/home/saptarshi/Dropbox (Heriot-Watt University Team)/RES_EPS_PathCad/datasets/pathcad_van_dataset/SighthillJunc/SighthillJunc_0/Navtech_Cartesian/000007.png');
radarImage8 = imread('/home/saptarshi/Dropbox (Heriot-Watt University Team)/RES_EPS_PathCad/datasets/pathcad_van_dataset/SighthillJunc/SighthillJunc_0/Navtech_Cartesian/000008.png');
radarImage9 = imread('/home/saptarshi/Dropbox (Heriot-Watt University Team)/RES_EPS_PathCad/datasets/pathcad_van_dataset/SighthillJunc/SighthillJunc_0/Navtech_Cartesian/000009.png');
radarImage10 = imread('/home/saptarshi/Dropbox (Heriot-Watt University Team)/RES_EPS_PathCad/datasets/pathcad_van_dataset/SighthillJunc/SighthillJunc_0/Navtech_Cartesian/000010.png');

radarImage = (radarImage1+radarImage2);

fusedImage = imfuse(radarImage,resizedMap, 'falsecolor');

% radarImage = (radarImage1+radarImage2+radarImage3+radarImage4+radarImage5+radarImage6+radarImage7+radarImage8+radarImage9+radarImage10);

figure(1)
% subplot(2,2,1)
imshow(radarImage)
figure(2)
% subplot(2,2,3)
imshow(resizedMap)
figure(3)
% subplot(2,2,[2 4])
imshow(fusedImage)



