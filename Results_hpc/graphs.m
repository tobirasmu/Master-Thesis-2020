%% For MNIST 
ranks = [2, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100, 150]
accs = [33.61, 52.20, 61.71, 67.55, 72.14, 79.41, 74.66, 74.89, 81.56, 75.15, 83.98, 83.65]
time = [1.481, 2.445, 3.114, 1.660, 1.906, 1.965, 2.397, 2.298, 3.005, 3.661, 4.796, 6.664]

figure()
set(gca, 'FontSize', 24)
yyaxis left
set(gca,'ycolor','b') 
plot(ranks, accs,'b', 'LineWidth',2)
hold on
plot(ranks, ones(1,12)*94.44,'b' , 'LineWidth',2)
ylabel('Accuracy')
yyaxis right
set(gca,'ycolor','r') 
plot(ranks, time, 'r', 'LineWidth',2)
hold on
plot(ranks, ones(1, 12)*1.7, 'r', 'LineWidth',2)
ylabel('Time for 10 forward pushes')
legend('Compressed', 'Full input', 'Compressed', 'Full input')
xlabel('Rank of the decomposition')

%% For THETIS
ranks = [30, 40, 50, 60, 70 ,80, 90, 100, 110, 120, 150, 200]
accs = [70, 70, 82, 78, 74, 76, 82, 82, 80, 88, 94, 94]
time = [30.53, 38.48, 47.79, 58.15, 72.01, 75.65, 90.57, 99.91, 106.21, 113.46, 138.88, 174.97]
figure()
set(gca, 'FontSize', 24)
yyaxis left
set(gca,'ycolor','b') 
plot(ranks, accs,'b', 'LineWidth',2)
hold on
plot(ranks, ones(1,12)*92,'b' , 'LineWidth',2)
ylabel('Accuracy')
yyaxis right
set(gca,'ycolor','r') 
plot(ranks, time, 'r', 'LineWidth',2)
hold on
plot(ranks, ones(1, 12)*95.24, 'r', 'LineWidth',2)
ylabel('Time for 10 forward pushes')
legend('Compressed', 'Full input', 'Compressed', 'Full input')
xlabel('Rank of the decomposition')