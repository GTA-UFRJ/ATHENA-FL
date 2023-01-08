
% CIFAR-10 Results
%path = 'RESULTS\SBRC_CIFAR_IID_OVA\';
%path = 'RESULTS\SBRC_CIFAR_2_CLASSES_OVA\';
%path = 'RESULTS\SBRC_CIFAR_5_CLASSES_OVA\';

% MNIST Results

%path = 'RESULTS\SBRC_MNIST_IID_OVA\';
%path = 'RESULTS\SBRC_MNIST_2_CLASSES_OVA\';
%path = 'RESULTS\SBRC_MNIST_5_CLASSES_OVA\';


tol = 0.001;
model_epoch = [];
model_acc = [];

for model_type = 0:1:9
model_type = 2;
    avg = csvread(path+"mean_model"+string(model_type));
    previous_value = avg(1);
    for i = 2:1:200
        if (previous_value*(1-tol) < avg(i) &&  previous_value*(1+tol) > avg(i))
            model_epoch = [model_epoch i];
            model_acc = [model_acc avg(i)];
            break
        else
            previous_value = avg(i);
        end
    end
end 

mean(model_epoch)
std(model_epoch)