% Plot Settings
labels = [];
color = {[0 0 255] [119 172 48] [80 80 80] [0 127 0] [0 114 189] [237 177 32] [255 0 0] [0 0 0] [128 128 128] [153 51 0]};
style = {'-' '-' '-.' '-' '-' '-' '-.' '-' '-.' '-'};
hold on;

%path = 'RESULTS\SBRC_CIFAR_IID_OVA\';

for i = 0:1:9
    mean = csvread(path+"mean_model"+string(i));
    inferior = csvread(path+"inferior_model"+string(i));
    superior = csvread(path+"superior_model"+string(i));
    axis = 1:length(mean);
    plot(axis,mean,style{i+1},'color',color{i+1}./255,'LineWidth',3)
    labels =  [labels 'Detector '+string(i)];   
end

legend(labels)
xlabel('Época')
ylabel('Acurácia (%)')
