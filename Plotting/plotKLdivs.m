clc
clear all
close all

%% load data
data_p3A = load('pointmass_3dof_ANN_shift_calc.mat');
data_p3K = load('pointmass_3dof_KNN_shift_calc.mat');
data_r3A = load('rigidbody_3dof_ANN_shift_calc.mat');
data_r3K = load('rigidbody_3dof_KNN_shift_calc.mat');
data_d6A = load('decoupled_6dof_ANN_shift_calc.mat');
data_d6K = load('decoupled_6dof_KNN_shift_calc.mat');
data_c6A = load('coupled_6dof_ANN_shift_calc.mat');
data_c6K = load('coupled_6dof_KNN_shift_calc.mat');

data_m6A = load('morecoupled_6dof_ANN_shift_calc.mat');
data_m6K = load('morecoupled_6dof_KNN_shift_calc.mat');


data_rb3ol = load('rb3_ol_test.mat');

%% Calculated Metrics
M_p3A = trapz(data_p3A.betas, data_p3A.D_KL_beta)
M_p3K = trapz(data_p3K.betas, data_p3K.D_KL_beta)
M_r3A = trapz(data_r3A.betas, data_r3A.D_KL_beta)
M_r3K = trapz(data_r3K.betas, data_r3K.D_KL_beta)
M_d6A = trapz(data_d6A.betas, data_d6A.D_KL_beta)
M_d6K = trapz(data_d6K.betas, data_d6K.D_KL_beta)
M_c6A = trapz(data_c6A.betas, data_c6A.D_KL_beta)
M_c6K = trapz(data_c6K.betas, data_c6K.D_KL_beta)
M_m6A = trapz(data_m6A.betas, data_m6A.D_KL_beta)
M_m6K = trapz(data_m6K.betas, data_m6K.D_KL_beta)


%% Plot
close all

fontsize = 18;


figure;
hold on
plot(data_p3A.betas, data_p3A.D_KL_beta,'b-x')
plot(data_p3K.betas, data_p3K.D_KL_beta,'b--x')
plot(data_r3A.betas, data_r3A.D_KL_beta,'r-x')
plot(data_r3K.betas, data_r3K.D_KL_beta,'r--x')
xlabel('$\beta$ [-]','Interpreter','latex','FontSize',fontsize)
ylabel('$D(P||Q_{\beta})$ [-]','Interpreter','latex','FontSize',fontsize)
legend('point-mass DNN','point-mass KNN','rigid-body DNN','rigid-body KNN')
ylim([-0.1 4])
saveas(gcf,'3dof_kldiv_plots.png')

figure;
hold on
plot(data_d6A.betas, [data_d6A.D_KL_beta(1:end-1),0],'b-x')
plot(data_d6K.betas, [data_d6K.D_KL_beta(1:end-1),0],'b--x')
plot(data_c6A.betas, data_c6A.D_KL_beta,'r-x')
plot(data_c6K.betas, data_c6K.D_KL_beta,'r--x')
plot(data_m6A.betas, data_m6A.D_KL_beta,'-x','Color',[0.4660 0.6740 0.1880])
plot(data_m6K.betas, data_m6K.D_KL_beta,'--x','Color',[0.4660 0.6740 0.1880])

xlabel('$\beta$ [-]','Interpreter','latex','FontSize',fontsize)
ylabel('$D(P||Q_{\beta})$ [-]','Interpreter','latex','FontSize',fontsize)
legend('decoupled 6DOF DNN','decoupled 6DOF KNN','coupled 6DOF DNN','coupled 6DOF KNN','coupled TVR 6DOF ANN', 'coupled TVR 6DOF KNN')
saveas(gcf,'6dof_kldiv_plots.png')


figure;
subplot(2,1,1)
hold on
plot(data_rb3ol.t_test(1:300,1),'b')
plot(data_rb3ol.yvis(:,1),'r--')
xlabel('Index [-]','Interpreter','latex','FontSize',fontsize)
ylabel('$u_1$ [N]','Interpreter','latex','FontSize',fontsize)
legend('Optimal','DNN')
subplot(2,1,2)
hold on
plot(data_rb3ol.t_test(1:300,2),'b')
plot(data_rb3ol.yvis(:,2),'r--')
xlabel('Index [-]','Interpreter','latex','FontSize',fontsize)
ylabel('$u_2$ [N]','Interpreter','latex','FontSize',fontsize)
saveas(gcf,'rb3dofol.png')
