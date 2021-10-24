clc
close all
clear all

% mnor grid alpha change....

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%NGSIM%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% Full Horizon Plot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % Round One
% roundOneAll = [0.14,0.22,0.28,0.34,0.41,0.49,0.58,0.67,0.77,0.88,0.99, ...
% 1.11,1.23,1.35,1.48,1.62,1.76,1.91,2.06,2.21,2.37, ...
% 2.54,2.71,2.89,3.07,3.26,3.45,3.65,3.85,4.06,4.27, ...
% 4.49,4.71,4.94,5.17,5.4,5.64,5.88,6.13,6.39,6.64, ...
% 6.9,7.17,7.44,7.71,7.99,8.27,8.55,8.84,9.14];
% roundOne = [roundOneAll(1), roundOneAll(10), roundOneAll(20), roundOneAll(30), roundOneAll(40), roundOneAll(50)];
% t1 = linspace(0,5,length(roundOne));
% 
% % Round Two
% roundTwoAll = [0.21,0.74,0.92,1.02,1.1,1.21,1.34,1.49,1.64,1.81,1.98,...
% 2.16,2.35,2.54,2.75,2.95,3.16,3.38,3.6,3.83,4.06,...
% 4.3,4.54,4.79,5.04,5.29,5.54,5.8,6.07,6.33,6.6,...
% 6.87,7.14,7.41,7.68,7.96,8.24,8.51,8.79,9.07,9.35,...
% 9.63,9.9,10.18,10.46,10.75,11.03,11.31,11.59,11.88];
% roundTwo = [roundTwoAll(1), roundTwoAll(10), roundTwoAll(20), roundTwoAll(30), roundTwoAll(40), roundTwoAll(50)];
% t2 = linspace(0,5,length(roundTwo));
% 
% % Round Three
% roundThreeAll = [0.24,0.24,0.27,0.33,0.39,0.44,0.49,0.53,0.58,0.63,0.68, ...
% 0.73,0.78,0.84,0.89,0.95,1.01,1.07,1.13,1.19,1.26, ...
% 1.32,1.39,1.45,1.52,1.59,1.66,1.73,1.81,1.88,1.95, ...
% 2.03,2.1,2.18,2.26,2.34,2.42,2.5,2.59,2.68,2.77, ...
% 2.86,2.96,3.05,3.15,3.26,3.37,3.48,3.59,3.71];
% roundThree = [roundThreeAll(1), roundThreeAll(10), roundThreeAll(20), roundThreeAll(30), roundThreeAll(40), roundThreeAll(50)];
% t3 = linspace(0,5,length(roundThree));
% 
% 
% % Round Four
% roundFourAll = [0.21,0.19,0.2,0.22,0.24,0.28,0.32,0.36,0.4,0.44,0.48, ...
% 0.52,0.56,0.61,0.66,0.71,0.76,0.82,0.88,0.94,1.0, ...
% 1.07,1.14,1.22,1.3,1.38,1.46,1.55,1.64,1.74,1.84, ...
% 1.94,2.04,2.15,2.27,2.38,2.5,2.62,2.74,2.87,3.01, ...
% 3.14,3.27,3.41,3.55,3.69,3.84,3.99,4.14,4.29];
% roundFour = [roundFourAll(1), roundFourAll(10), roundFourAll(20), roundFourAll(30), roundFourAll(40), roundFourAll(50)];
% t4 = linspace(0,5,length(roundFour));
% 
% 
% % Round Five
% roundFiveAll = [0.17,0.16,0.16,0.18,0.19,0.21,0.24,0.27,0.3,0.33,0.37, ...
% 0.4,0.44,0.48,0.51,0.56,0.6,0.65,0.7,0.75,0.81, ...
% 0.86,0.92,0.99,1.05,1.12,1.19,1.26,1.34,1.42,1.5, ...
% 1.58,1.66,1.75,1.83,1.92,2.0,2.09,2.18,2.27,2.36, ...
% 2.46,2.55,2.65,2.75,2.86,2.96,3.07,3.19,3.3];
% roundFive = [roundFiveAll(1), roundFiveAll(10), roundFiveAll(20), roundFiveAll(30), roundFiveAll(40), roundFiveAll(50)];
% t5 = linspace(0,5,length(roundFive));
% 
% % Round Six
% roundSixAll = [0.15,0.14,0.14,0.15,0.16,0.17,0.19,0.21,0.22,0.25,0.27, ...
% 0.29,0.32,0.35,0.37,0.4,0.43,0.47,0.5,0.54,0.58, ...
% 0.63,0.68,0.73,0.78,0.84,0.9,0.96,1.03,1.1,1.17, ...
% 1.25,1.32,1.4,1.49,1.57,1.66,1.75,1.84,1.94,2.03, ...
% 2.13,2.24,2.34,2.45,2.56,2.67,2.79,2.91,3.03];
% roundSix = [roundSixAll(1), roundSixAll(10), roundSixAll(20), roundSixAll(30), roundSixAll(40), roundSixAll(50)];
% t6 = linspace(0,5,length(roundSix));
% 
% % Round Seven
% roundSevenAll = [0.13,0.12,0.12,0.12,0.13,0.15,0.16,0.17,0.19,0.2,0.22, ...
% 0.24,0.26,0.28,0.3,0.32,0.35,0.38,0.4,0.43,0.47, ...
% 0.5,0.54,0.58,0.62,0.66,0.71,0.76,0.82,0.87,0.93, ...
% 0.99,1.06,1.12,1.19,1.26,1.34,1.41,1.49,1.57,1.65, ...
% 1.73,1.81,1.9,1.99,2.08,2.18,2.28,2.38,2.49];
% roundSeven = [roundSevenAll(1), roundSevenAll(10), roundSevenAll(20), roundSevenAll(30), roundSevenAll(40), roundSevenAll(50)];
% t7 = linspace(0,5,length(roundSeven));
% 
% 
% 
% % plots
% plot(t1,roundOne, '-^', 'LineWidth',2, 'MarkerSize',12)
% hold on
% plot(t2,roundTwo, '-h', 'LineWidth',2, 'MarkerSize',12)
% hold on
% plot(t3,roundThree, '-*', 'LineWidth',2, 'MarkerSize',12)
% hold on
% plot(t4,roundFour, '-o', 'LineWidth',2, 'MarkerSize',12)
% hold on
% plot(t5,roundFive, '-d', 'LineWidth',2, 'MarkerSize',12)
% hold on
% plot(t6,roundSix, '-p', 'LineWidth',2, 'MarkerSize',12)
% hold on
% plot(t7,roundSeven, '-s', 'LineWidth',2, 'MarkerSize',12)
% hold on
% 
% % Plot params
% xlabel('Time (Sec)', 'FontSize', 15);
% ylabel('RMS Error (m)', 'FontSize', 15);
% % lgd = legend('CV', 'Network', 'PFilter(Pose+Motion)', 'PFilter(Pose+TrueMotion)', 'PFilter(Pose+Motion+RegenParticles)');
% lgd = legend('Round-One', 'Round-Two', 'Round-Three', 'Round-Four', 'Round-Five', 'Round-Six', 'Round-Seven');
% lgd.FontSize = 13;
% a = get(gca,'XTickLabel');  
% set(gca,'XTickLabel',a,'fontsize',15)
% a = get(gca,'YTickLabel');  
% set(gca,'YTickLabel',a,'fontsize',15)
% grid on
% ax=gca;
% ax.GridAlpha=0.95;
% grid minor
% ax=gca;
% ax.GridAlpha=0.95;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% Only End Horizon Plot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% End Horizon Plot agaist rounds
endHorizonAgaistRoundsTrain = [13.29, 4.68, 4.74, 4.35, 3.86, 3.46];
endHorizonAgaistRoundsVal = [15.25, 4.37, 4.98, 4.03, 3.97, 3.51];
GTSurrounding = [2.98,2.98,2.98,2.98,2.98,2.98];
rounds = [1,2,3,4,5,6];
% plots

plot(rounds,endHorizonAgaistRoundsTrain, '-s', 'LineWidth',2, 'MarkerSize',12)
hold on
plot(rounds,endHorizonAgaistRoundsVal, '-*', 'LineWidth',2, 'MarkerSize',12)
hold on
plot(rounds,GTSurrounding, '-*', 'LineWidth',2, 'MarkerSize',12, 'color', 'g')
hold on 

% Plot params
xlabel('Re-Train Rounds', 'FontSize', 20);
ylabel('RMS Error at 5th Sec (50th frame)', 'FontSize', 20);
title('Retrain-Technique (New Schedule Sampling');
lgd = legend('Train Data (I-80 4.15 PM)', 'Val Data (I-80 5.15 PM)', 'GT Surrounding');
lgd.FontSize = 15;
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',15)
a = get(gca,'YTickLabel');  
set(gca,'YTickLabel',a,'fontsize',15)
grid on
ax=gca;
ax.GridAlpha=0.95;
grid minor
ax=gca;
ax.GridAlpha=0.95;