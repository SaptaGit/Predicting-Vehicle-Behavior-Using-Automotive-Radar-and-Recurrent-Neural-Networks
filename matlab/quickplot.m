close all
clc
clear all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% Alvie Junction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


positionNoAll = [0.66109137 0.85063599 1.00782465 1.14035895 1.29940116 1.43850732 ...
 1.55490811 1.67937102 1.79415096 1.90716601 2.05145159 2.17123321 ...
 2.33910103 2.49422916 2.65447473 2.7864605  2.98857303 3.157593 ...
 3.33766824 3.56861762];
positionNo = [positionNoAll(1), positionNoAll(4), positionNoAll(8), positionNoAll(12), positionNoAll(16),positionNoAll(20)];
t1 = linspace(0,5,length(positionNo));


position10All = [0.65617333 0.80775407 0.97426038 1.10523692 1.26501868 1.36479992 ...
 1.52301247 1.62638544 1.75948207 1.91113464 2.04052442 2.18311328 ...
 2.3528716  2.52916267 2.71657772 2.8431709  3.04987708 3.24426856 ...
 3.44599434 3.68288194];

position10 = [position10All(1), position10All(4), position10All(8), position10All(12), position10All(16),position10All(20)];
t2 = linspace(0,5,length(position10));


position20All = [0.63870072 0.84481584 1.00336639 1.14687779 1.28992286 1.39508602 ...
 1.53691281 1.65668628 1.77993745 1.91113551 2.05663998 2.23195372 ...
 2.40886744 2.56505056 2.73433484 2.90542316 3.07388848 3.28204131 ...
 3.47916942 3.71688051];

position20 = [position20All(1), position20All(4), position20All(8), position20All(12), position20All(16),position20All(20)];
t3 = linspace(0,5,length(position20));


position30All = [0.69741486 0.90088338 1.06206547 1.2237492  1.35517542 1.48318714 ...
 1.61577428 1.75675313 1.91141043 2.04551522 2.18734216 2.3262087 ...
 2.48532822 2.65221064 2.81365637 2.99414896 3.16379255 3.36585618 ...
 3.59424916 3.84901667];

position30 = [position30All(1), position30All(4), position30All(8), position30All(12), position30All(16),position30All(20)];
t4 = linspace(0,5,length(position30));



% plots
plot(t1,positionNo, '-*', 'LineWidth',2, 'MarkerSize',12)
hold on
plot(t2,position10, '-s', 'LineWidth',2, 'MarkerSize',12)
hold on
plot(t3,position20, '-o', 'LineWidth',2, 'MarkerSize',12)
hold on
plot(t4,position30, '-p', 'LineWidth',2, 'MarkerSize',12)
hold on

% Plot params
xlabel('Time (Sec)', 'FontSize', 15);
ylabel('RMS Error (m)', 'FontSize', 15);
legend('No Misdetection', '10% Misdetection','20% Misdetection', '30% Misdetection')
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







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% Sighthill Junction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
close all

positionNoAll = [0.71539388 1.2867068  1.06759797 1.26651194 1.26128427 1.37779653 ...
 1.40906187 1.7604538  1.77556281 2.06707725 2.27156388 2.28042668 ...
 2.46844399 2.8296784  2.85016934 3.01818912 3.30941444 3.45851871 ...
 3.72502597 3.78928181];
positionNo = [positionNoAll(1), positionNoAll(4), positionNoAll(8), positionNoAll(12), positionNoAll(16),positionNoAll(20)];
t1 = linspace(0,5,length(positionNo));


position10All = [0.83073351 1.30230387 1.33244396 1.32775199 1.32097675 1.59329765 ...
 1.55882071 1.74207001 1.98838042 2.1041492  2.10641577 2.32290452 ...
 2.49179591 2.64663557 2.92464365 3.11186221 3.28752962 3.3758052 ...
 3.78550999 3.95442185];

position10 = [position10All(1), position10All(4), position10All(8), position10All(12), position10All(16),position10All(20)];
t2 = linspace(0,5,length(position10));


position20All = [0.9735524 1.32748032 1.22196528 1.30002863 1.53364172 1.42157152 ...
 1.43902813 1.82667098 1.74451777 1.84668055 2.11639137 2.34435163 ...
 2.3838301  2.9684699  2.95173359 3.10031168 3.29434649 3.54848122 ...
 3.79909456 4.06763429];

position20 = [position20All(1), position20All(4), position20All(8), position20All(12), position20All(16),position20All(20)];
t3 = linspace(0,5,length(position20));


position30All = [1.14180136 1.24917887 1.15995807 1.482820232 1.56888551 1.47957995 ...
 1.70756766 2.02016839 2.05385965 2.1658719  2.44438084 2.57743938 ...
 2.70635017 2.94154847 3.16163101 3.32513813 3.67523788 3.74511201 ...
 4.23447989 4.27439926];

position30 = [position30All(1), position30All(4), position30All(8), position30All(12), position30All(16),position30All(20)];
t4 = linspace(0,5,length(position30));



% plots
plot(t1,positionNo, '-*', 'LineWidth',2, 'MarkerSize',12)
hold on
plot(t2,position10, '-s', 'LineWidth',2, 'MarkerSize',12)
hold on
plot(t3,position20, '-o', 'LineWidth',2, 'MarkerSize',12)
hold on
plot(t4,position30, '-p', 'LineWidth',2, 'MarkerSize',12)
hold on

% Plot params
xlabel('Time (Sec)', 'FontSize', 15);
ylabel('RMS Error (m)', 'FontSize', 15);
legend('No Misdetection', '10% Misdetection','20% Misdetection', '30% Misdetection')
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