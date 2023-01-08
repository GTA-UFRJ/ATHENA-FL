% Data
net_size = [9560864 13215768 83935320 563760]./(2^(10));
names = {'MobileNetV2' 'MobileNet' 'Xcepetion' 'Detector'};

% Plot 
bar([1:4],net_size);
set(gca,'xticklabel',names);
ylabel('Tamanho do Modelo (kB)')