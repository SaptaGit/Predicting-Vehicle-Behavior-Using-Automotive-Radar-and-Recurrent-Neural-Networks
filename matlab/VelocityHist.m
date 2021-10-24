clc
clear all
close all

cellResolution = 0.173611; 
velocityProfile = importdata('/home/saptarshi/PythonCode/Junction/infos/smoothText.txt');
highestFrame = 65; % check the end the file to find out
profileLength = length(velocityProfile)-1; % To ignore the higest frame count at the end
for idx=1:profileLength
    currentProfile = velocityProfile(idx);
    splittedStr = split(currentProfile, ',');
    splittedStrLen = length(splittedStr)-1; % To ignore the manuever info at the end and blank str due to comma
    floatProfile = zeros(1,splittedStrLen);
    for jdx=1:splittedStrLen
        floatProfile(jdx) = str2double(splittedStr(jdx));
    end
    % Convert pixel to meter per sec
    floatProfile = floatProfile*cellResolution;
    maneuverInfo = splittedStr(splittedStrLen+1);
    lengthDiff = highestFrame-splittedStrLen+1;
    xAxisVal = lengthDiff:highestFrame;
    
    if strcmp(maneuverInfo,'Straight')
        plot(xAxisVal,floatProfile,'color','g','linewidth',3, 'linestyle', '--')
        hold on
    elseif strcmp(maneuverInfo,'Turn')
        plot(xAxisVal,floatProfile,'color','r','linewidth',3, 'linestyle', '-')
        hold on
    else
        disp('Unknown man')
    end
    
end

xlabel('Distance from junction (m)','FontSize',18);
ylabel('Velocity (m/s)','FontSize',18);
legend({'Straight', 'Turn'}, 'FontSize',20)

a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'fontsize',18)
xlim([0 66])

xticks([0 15 35 50 65])
xticklabels({'-100','-75','-50','-25','0'})

grid on
grid minor


% % % % % % avgVelocity = importdata('/home/saptarshi/PythonCode/Junction/AverageVelocity.txt');
% % % % % % straightArray = {};
% % % % % % turnArray = {};
% % % % % % 
% % % % % % for idx=1:length(avgVelocity)
% % % % % %     currentCell = char(string(avgVelocity(idx)));
% % % % % %     currentInfo = split(currentCell(2:end-1),',');
% % % % % %     maneuver = char(currentInfo(1));
% % % % % %     maneuver = maneuver(2:end-1);
% % % % % %     velocity = str2double(currentInfo(2));
% % % % % %     if strcmp(maneuver,'Straight')
% % % % % %         straightArray = [straightArray, velocity];
% % % % % %     else
% % % % % %         turnArray = [turnArray, velocity];
% % % % % %     end
% % % % % % end
% % % % % % 
% % % % % % H1 = hist(cell2mat(straightArray), 30);
% % % % % % H2 = hist(cell2mat(turnArray), 30);
% % % % % % plot(H1)
% % % % % % hold on
% % % % % % plot(H2)
% % % % % % 
% % % % % % edges=linspace(1,160,31);         % pick number of bins, points is 1+ that over your range
% % % % % % N = histcounts(cell2mat(straightArray),edges);          % get the counts in those bins
% % % % % % x=filter(edges,[0.5 0.5],1);      % midpoint of bins; mean of edges
% % % % % % plot(x(2:end),N)                  % and plot...N.B. start with second x to get number bins wanted



