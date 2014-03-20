%This Matlab script can be used to generate Figure 2 in the article:
%
%Emil Björnson, Björn Ottersten, "A Framework for Training-Based Estimation
%in Arbitrarily Correlated Rician MIMO Channels with Rician Disturbance,”
%IEEE Transactions on Signal Processing, vol. 58, no. 3, pp. 1807-1820,
%March 2010.
%
%Download article: http://kth.diva-portal.org/smash/get/diva2:337243/FULLTEXT01
%
%This is version 1.0 (Last edited: 2014-03-20)
%
%License: This code is licensed under the GPLv2 license. If you in any way
%use this code for research that results in publications, please cite our
%original article listed above.
%
%Please note that the channels are generated randomly, thus the results
%will not be exactly the same as in the paper.

%Initialization
close all;
clear all;


%%Simulation parameters

rng('shuffle'); %Initiate the random number generators with a random seed
%%If rng('shuffle'); is not supported by your Matlab version, you can use
%%the following commands instead:
%randn('state',sum(100*clock));


%Number of realizations in the Monte-Carlo simulations
nbrOfMonteCarloRealizations = 100000;

Nt = 5; %Number of transmit antennas and length of the training sequence
Nr = 5; %Number of receive antennas

%Range of total training power in the simulation
totalTrainingPower_dB = 0:1:20; %In decibel scale
totalTrainingPower = 10.^(totalTrainingPower_dB/10); %In linear scale

%Options in the optimization Matlab algorithms used in this simulations
option = optimset('Display','off','TolFun',1e-7,'TolCon',1e-7,'Algorithm','interior-point');


%Placeholders for storing the MSEs with different channel estimators.
%Second dimension is 1 for the theoretical MSE results and 2 for empirical
%results using Monte-Carlo simulations
average_MSE_MMSE_estimator_optimal = zeros(length(totalTrainingPower),2); %MMSE estimator with optimal training
average_MSE_MMSE_estimator_heuristic = zeros(length(totalTrainingPower),2); %MMSE estimator with heuristic training (from Heuristic 1)
average_MSE_MMSE_estimator_uniform = zeros(length(totalTrainingPower),2); %MMSE estimator with a diagonal training matrix
average_MSE_onesided_estimator = zeros(length(totalTrainingPower),2); %One-sided linear estimator (from [8] by M. Biguesh and A. Gershman)


%A particular coupling matrix from Eq. (28) in [29].
V=25/5.7*[0.1 0 1 0 0; 0 0.1 1 0 0; 0 0 1 0 0; 0 0 1 0.25 0; 0 0 1 0 0.25];

%Compute the covariance matrix for the given coupling matrix
R = diag(V(:));
Rsqrt = diag(sqrt(V(:))); %Square root of covariance matrix

R_T = diag(sum(V,1)); %Compute transmit-side covariance matrix under the Weichselberger model
R_R = diag(sum(V,2)); %Compute receive-side covariance matrix under the Weichselberger model


%Compute optimal training power allocation for the MMSE estimator using built-in optimization algorithms
trainingpower_MMSE_optimal = zeros(Nt,length(totalTrainingPower)); %Vector with power allocation for each total training power

for k = 1:length(totalTrainingPower) %Go through each total training power
    trainingpower_initial = totalTrainingPower(k)*ones(Nt,1)/Nt; %Equal power allocation is initial point in optimization
    
    %Optimize power allocation using fmincon (i.e., minimize MSE under
    %total training power constraint and all powers are nonnegative.
    trainingpower_MMSE_optimal(:,k) = fmincon(@(q) functionMSEmatrix(R,q,Nr),trainingpower_initial,ones(1,Nt),totalTrainingPower(k),[],[],zeros(Nt,1),totalTrainingPower(k)*ones(Nt,1),[],option);
end


%Compute heuristic training power allocating according to Heuristic 1
%in the paper, which uses Corollary 1 and Eq. (14) heuristically
[eigenvalues_sorted,permutationorder] = sort(diag(R_T),'descend'); %Compute and sort the eigenvalues
[~,inversePermutation] = sort(permutationorder); %Keep track of the order of eigenvalues

q_MMSE_heuristic = zeros(Nt,length(totalTrainingPower));
for k = 1:length(totalTrainingPower) %Go through each total training power
    alpha_candidates = (totalTrainingPower(k)+cumsum(1./eigenvalues_sorted(1:Nt,1)))./(1:Nt)'; %Compute different values on the Lagrange multiplier in Eq. (14), given that 1,2,...,Nt of the eigendirections get non-zero power
    optimalIndex = find(alpha_candidates-1./eigenvalues_sorted(1:Nt,1)>0 & alpha_candidates-[1./eigenvalues_sorted(2:end,1); Inf]<0); %Find the true Lagrange multiplier alpha by checking which one of the candidates that only turns on the eigendirections that are supposed to be on
    q_MMSE_heuristic(:,k) = max([alpha_candidates(optimalIndex)-1./eigenvalues_sorted(1:Nt,1) zeros(Nt,1)],[],2); %Compute the power allocation according to  Eq. (14) using the optimal alpha
end

q_MMSE_heuristic = q_MMSE_heuristic(inversePermutation,:); %Finalize the heuristic power allocation by reverting the sorting of the eigenvalues


%Compute uniform power allocation for different total training powers
q_uniform = (ones(Nt,1)/Nt)*totalTrainingPower;


%Initialization of Monte-Carlo simulations
vecH_realizations = Rsqrt*( randn(Nt*Nr,nbrOfMonteCarloRealizations)+1i*randn(Nt*Nr,nbrOfMonteCarloRealizations) ) / sqrt(2); %Generate channel realizations on vectorized form. Each column is an independent realization of vec(H).
vecN_realizations = ( randn(Nt*Nr,nbrOfMonteCarloRealizations)+1i*randn(Nt*Nr,nbrOfMonteCarloRealizations) ) / sqrt(2); %Generate noise realizations on vectorized form. Each column is an independent realization of vec(N).


%Compute MSEs for each estimator and total training power
for k = 1:length(totalTrainingPower)
    
    %MMSE estimator: Optimal training power allocation
    P_tilde = kron(diag(sqrt(trainingpower_MMSE_optimal(:,k))),eye(Nr)); %Compute effective training matrix as in Eq. (5)
    
    average_MSE_MMSE_estimator_optimal(k,1) = trace(R - (R*P_tilde'/(P_tilde*R*P_tilde' + eye(length(R))))*P_tilde*R); %Compute the MSE using Eq. (8)
    
    H_hat = (R*P_tilde'/(P_tilde*R*P_tilde'+eye(length(R)))) * (P_tilde*vecH_realizations+vecN_realizations); %Compute the estimate by Monte-Carlo simulations using Eq. (6)
    average_MSE_MMSE_estimator_optimal(k,2) = mean( sum(abs(vecH_realizations - H_hat).^2,1) ); %Compute the MSE by Monte-Carlo simulations using the MSE definition
    
    
    %MMSE estimator: Heuristic training power allocation
    P_tilde = kron(diag(sqrt(q_MMSE_heuristic(:,k))),eye(Nr));  %Compute effective training matrix as in Eq. (5)
    
    average_MSE_MMSE_estimator_heuristic(k,1) = trace(R - (R*P_tilde'/(P_tilde*R*P_tilde' + eye(length(R))))*P_tilde*R); %Compute the MSE using Eq. (8)
    
    H_hat = (R*P_tilde'/(P_tilde*R*P_tilde'+eye(length(R)))) * (P_tilde*vecH_realizations + vecN_realizations); %Compute the estimate by Monte-Carlo simulations using Eq. (6)
    average_MSE_MMSE_estimator_heuristic(k,2) = mean( sum(abs(vecH_realizations - H_hat).^2,1) ); %Compute the MSE by Monte-Carlo simulations using the MSE definition
    
    
    %MMSE estimator: Uniform training power allocation
    P_tilde = kron(sqrt(totalTrainingPower(k))*eye(Nt)/sqrt(Nt),eye(Nr));  %Compute effective training matrix as in Eq. (5)
    
    average_MSE_MMSE_estimator_uniform(k,1) = trace(R - (R*P_tilde'/(P_tilde*R*P_tilde' + eye(length(R))))*P_tilde*R); %Compute the MSE using Eq. (8)
    
    H_hat = (R*P_tilde'/(P_tilde*R*P_tilde'+eye(length(R)))) * (P_tilde*vecH_realizations + vecN_realizations); %Compute the estimate by Monte-Carlo simulations using Eq. (6)
    average_MSE_MMSE_estimator_uniform(k,2) = mean( sum(abs(vecH_realizations - H_hat).^2,1) ); %Compute the MSE by Monte-Carlo simulations using the MSE definition
    
    %One-sided linear estimator: Optimal training power allocation
    %
    %This heuristic estimator originates from the following paper:
    %
    %[8] M. Biguesh and A. Gershman, "Training-based MIMO channel estimation:
    %A study of estimator tradeoffs and optimal training signals,"
    %IEEE Trans. Signal Process., vol. 54, no. 3, pp. 884-893, 2006.
    %
    %Note: this estimator is called "LMMSE estimator" in [8], but this
    %in a incorrect use of terminology.
    
    P_training = diag(sqrt(q_MMSE_heuristic(:,k))); %Compute training matrix using the optimal power allocation in Eq. (41) of [8], which actually is the same as Heuristic 1 in this paper.
    P_tilde = kron(P_training,eye(Nr)); %Compute effective training matrix as in Eq. (5)
    
    average_MSE_onesided_estimator(k,1) = trace(inv(inv(R_T)+P_training*P_training'/Nr)); %Compute the MSE using Eq. (34) in [8]
    
    Ao = (P_training'*R_T*P_training + Nr*eye(Nt))\P_training'*R_T; %Compute the matrix Ao in the one-sided linear estimator using Eq. (25) in [8]
    H_hat = kron(transpose(Ao),eye(Nr))*(P_tilde*vecH_realizations + vecN_realizations); %Compute the estimate by Monte-Carlo simulations using Eq. (26) in [8]
    average_MSE_onesided_estimator(k,2) = mean( sum(abs(vecH_realizations - H_hat).^2,1) );  %Compute the MSE by Monte-Carlo simulations using the MSE definition
    
end


%Select a subset of training power for which we will plot markers
subset = linspace(1,length(totalTrainingPower_dB),6);
subset = subset(2:end-1);

normalizationFactor = trace(R); %Set MSE normalization factor to trace(R), so that the figures show normalized MSEs from 0 to 1.


%Plot the numerical results using the theoretical MSE formulas
figure(1); hold on; box on;

plot(totalTrainingPower_dB,average_MSE_MMSE_estimator_uniform(:,1)/normalizationFactor,'k--','LineWidth',1);
plot(totalTrainingPower_dB,average_MSE_onesided_estimator(:,1)/normalizationFactor,'r-','LineWidth',1);
plot(totalTrainingPower_dB(subset(1)),average_MSE_MMSE_estimator_heuristic(subset(1),1)/normalizationFactor,'b+-.','LineWidth',1);
plot(totalTrainingPower_dB(subset(1)),average_MSE_MMSE_estimator_optimal(subset(1),1)/normalizationFactor,'ko-','LineWidth',1);

legend('MMSE, uniform','One-sided linear, optimal','MMSE, heuristic','MMSE, optimal','Location','SouthWest')

plot(totalTrainingPower_dB(subset),average_MSE_MMSE_estimator_optimal(subset,1)/normalizationFactor,'ko','LineWidth',1);
plot(totalTrainingPower_dB(subset),average_MSE_MMSE_estimator_heuristic(subset,1)/normalizationFactor,'b+','LineWidth',1);
plot(totalTrainingPower_dB,average_MSE_MMSE_estimator_optimal(:,1)/normalizationFactor,'k','LineWidth',1);
plot(totalTrainingPower_dB,average_MSE_MMSE_estimator_heuristic(:,1)/normalizationFactor,'b-.','LineWidth',1);

set(gca,'YScale','Log'); %Set log-scale on vertical axis
xlabel('Total Training Power (dB)');
ylabel('Normalized MSE');
axis([0 20 1e-2 1]);

%Add title to Figure 1 to differentiate from Figure 2
title('Results based on theoretical formulas');


%Plot the numerical results using Monte-Carlo simulations
figure(2); hold on; box on;

plot(totalTrainingPower_dB,average_MSE_MMSE_estimator_uniform(:,2)/normalizationFactor,'k--','LineWidth',1);
plot(totalTrainingPower_dB,average_MSE_onesided_estimator(:,2)/normalizationFactor,'r-','LineWidth',1);
plot(totalTrainingPower_dB(subset(1)),average_MSE_MMSE_estimator_heuristic(subset(1),2)/normalizationFactor,'b+-.','LineWidth',1);
plot(totalTrainingPower_dB(subset(1)),average_MSE_MMSE_estimator_optimal(subset(1),2)/normalizationFactor,'ko-','LineWidth',1);

legend('MMSE, uniform','One-sided linear, optimal','MMSE, heuristic','MMSE, optimal','Location','SouthWest')

plot(totalTrainingPower_dB(subset),average_MSE_MMSE_estimator_optimal(subset,2)/normalizationFactor,'ko','LineWidth',1);
plot(totalTrainingPower_dB(subset),average_MSE_MMSE_estimator_heuristic(subset,2)/normalizationFactor,'b+','LineWidth',1);
plot(totalTrainingPower_dB,average_MSE_MMSE_estimator_optimal(:,2)/normalizationFactor,'k','LineWidth',1);
plot(totalTrainingPower_dB,average_MSE_MMSE_estimator_heuristic(:,2)/normalizationFactor,'b-.','LineWidth',1);

set(gca,'YScale','Log'); %Set log-scale on vertical axis
xlabel('Total Training Power (dB)');
ylabel('Normalized MSE');
axis([0 20 1e-2 1]);

%Add title to Figure 2 to differentiate from Figure 1
title('Results based on Monte-Carlo simulations');
