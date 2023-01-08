% CIFAR-10

% duas amostras por cliente
%acc = [ 40 30];
%err = [5 3];

% cinco amostras por cliente
%acc = [ 66 70];
%err = [ 2 10 ];

% IID
acc = [ 99.3 99.8];
err = [ 0.1 0.1];

% MNIST

% duas amostras por cliente
%acc = [ 20 47];
%err = [10 1];

% cinco amostras por cliente
%acc = [ 64 76.9 ];
%err = [ 10 0.5];

% IID
%acc = [ 66 70];
%err = [ 2 10 ];


%names = {'Aprendizado Federado Tradicional' 'ATHENA-FL'};% 'FedOVA'};
names = {'        Aprendizado \newline Federado Tradicional' 'ATHENA-FL'};

bar([1:2],acc);
set(gca,'xticklabel',names);

hold on;

er = errorbar([1:2],acc(1:2),err,err);    
er.Color = [0 0 0];                            
er.LineStyle = 'none'; 

ylabel('Acurácia (%)');