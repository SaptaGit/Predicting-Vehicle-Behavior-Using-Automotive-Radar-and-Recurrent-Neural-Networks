clear all
close all
clc

maneuverHistData = textread('/home/saptarshi/PythonCode/Junction/infos/errorHistManeuver.txt','%s', 'delimiter','\n','whitespace','');
correctFlag = 0;
wrongFlag = 0;
for idx = 1 : length(maneuverHistData)
    currentErrorStr = convertStringsToChars(string(maneuverHistData(idx)));
    currentErrorVal = split(currentErrorStr(2:end-1), ',');
    errorFloat = str2double(char(currentErrorVal(1)));
    categoryFloat = str2double(char(currentErrorVal(2)));
    if categoryFloat == 1
        if correctFlag == 0
            correctError =  errorFloat ;
            correctFlag = 1;
        else
            correctError = [correctError, errorFloat];
        end
        
    elseif categoryFloat == 0
        if wrongFlag == 0
            wrongError =  errorFloat ;
            wrongFlag = 1;
        else
            wrongError = [wrongError, errorFloat];
        end
    else
        disp('Unknown categry value')
    end
end
% 
% % Generate the histgrams
% [correntCounts, correntBinCentres] = hist(correctError, 100);
% [wrongCounts, wrongBinCentres] = hist(wrongError, 100);
% 
% plot(correntCounts, correntBinCentres, 'g-');
% hold on
% plot(wrongCounts, wrongBinCentres, 'r-');
% legend({'Correct', 'Wrong'});

histogram(correctError,200, 'FaceColor',[1 1 0])
hold on
histogram(wrongError,200, 'FaceColor',[1 0 0])
xlabel('Error (m)', 'FontSize', 20);
ylabel('Sample Count', 'FontSize', 20);
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',20)
a = get(gca,'YTickLabel');  
set(gca,'YTickLabel',a,'fontsize',20)
lgd = legend({'True Maneuver', 'False Maneuver'});
lgd.FontSize = 25;



disp('done')