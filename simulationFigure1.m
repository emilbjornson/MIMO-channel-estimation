%This Matlab script can be used to generate Figure 1 in the article:
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
nbrOfMonteCarloRealizations = 1000;

nbrOfCouplingMatrices = 50; %Number of coupling matrices that the results are averaged over (it equals 5000 in the paper)

Nt = 8; %Number of transmit antennas and length of the training sequence
Nr = 4; %Number of receive antennas

%Range of total training power in the simulation
totalTrainingPower_dB = 0:1:20; %In decibel scale
totalTrainingPower = 10.^(totalTrainingPower_dB/10); %In linear scale

%Options in the optimization Matlab algorithms used in this simulations
option = optimset('Display','off','TolFun',1e-7,'TolCon',1e-7,'Algorithm','interior-point');


%Placeholders for storing the MSEs with different channel estimators. Third
%dimension is 1 for the theoretical MSE results and 2 for empirical results
%using Monte-Carlo simulations
average_MSE_MMSE_estimator_optimal = zeros(length(totalTrainingPower),nbrOfCouplingMatrices,2); %MMSE estimator with optimal training
average_MSE_MMSE_estimator_heuristic = zeros(length(totalTrainingPower),nbrOfCouplingMatrices,2); %MMSE estimator with heuristic training (from Heuristic 1)
average_MSE_MVU_estimator = zeros(length(totalTrainingPower),nbrOfCouplingMatrices,2); %MVU estimator with optimal training (uniform training)
average_MSE_onesided_estimator = zeros(length(totalTrainingPower),nbrOfCouplingMatrices,2); %One-sided linear estimator (from [8] by M. Biguesh and A. Gershman)
average_MSE_twosided_estimator = zeros(length(totalTrainingPower),nbrOfCouplingMatrices,2); %Two-sided linear estimator (from [27] by D. Katselis, E. Kofidis, and S. Theodoridis)


%Iteration over each realization of the random channel statistics
for statisticsIndex = 1:nbrOfCouplingMatrices
    
    %Generate coupling matrix V of the Weichselberger model with
    %chi-squared distributed variables (with 2 degrees of freedom).
    V = abs(randn(Nr,Nt)+1i*randn(Nr,Nt)).^2;
    V = Nt*Nr*V/sum(V(:)); %Normalize the Frobenius norm of V to Nt x Nr.
    
    %Compute the covariance matrix for the given coupling matrix
    R = diag(V(:));
    
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
    vecH_realizations = sqrtm(R)*( randn(Nt*Nr,nbrOfMonteCarloRealizations)+1i*randn(Nt*Nr,nbrOfMonteCarloRealizations) ) / sqrt(2); %Generate channel realizations on vectorized form. Each column is an independent realization of vec(H).
    vecN_realizations = ( randn(Nt*Nr,nbrOfMonteCarloRealizations)+1i*randn(Nt*Nr,nbrOfMonteCarloRealizations) ) / sqrt(2); %Generate noise realizations on vectorized form. Each column is an independent realization of vec(N).
    
    
    %Compute MSEs for each estimator and total training power
    for k = 1:length(totalTrainingPower)
        
        %MMSE estimator: Optimal training power allocation
        P_tilde = kron(diag(sqrt(trainingpower_MMSE_optimal(:,k))),eye(Nr)); %Compute effective training matrix as in Eq. (5)
        
        average_MSE_MMSE_estimator_optimal(k,statisticsIndex,1) = trace(R - (R*P_tilde'/(P_tilde*R*P_tilde' + eye(length(R))))*P_tilde*R); %Compute the MSE using Eq. (8)
        
        H_hat = (R*P_tilde'/(P_tilde*R*P_tilde'+eye(length(R)))) * (P_tilde*vecH_realizations+vecN_realizations); %Compute the estimate by Monte-Carlo simulations using Eq. (6)
        average_MSE_MMSE_estimator_optimal(k,statisticsIndex,2) = mean( sum(abs(vecH_realizations - H_hat).^2,1) ); %Compute the MSE by Monte-Carlo simulations using the MSE definition
        
        
        %MMSE estimator: Heuristic training power allocation
        P_tilde = kron(diag(sqrt(q_MMSE_heuristic(:,k))),eye(Nr));  %Compute effective training matrix as in Eq. (5)
        
        average_MSE_MMSE_estimator_heuristic(k,statisticsIndex,1) = trace(R - (R*P_tilde'/(P_tilde*R*P_tilde' + eye(length(R))))*P_tilde*R); %Compute the MSE using Eq. (8)
        
        H_hat = (R*P_tilde'/(P_tilde*R*P_tilde'+eye(length(R)))) * (P_tilde*vecH_realizations + vecN_realizations); %Compute the estimate by Monte-Carlo simulations using Eq. (6)
        average_MSE_MMSE_estimator_heuristic(k,statisticsIndex,2) = mean( sum(abs(vecH_realizations - H_hat).^2,1) ); %Compute the MSE by Monte-Carlo simulations using the MSE definition
        
        
        %MVU estimator: Optimal uniform training power allocation
        P_training = diag(sqrt(q_uniform(:,k))); %Uniform power allocation
        P_tilde = kron(transpose(P_training),eye(Nr));  %Compute effective training matrix as in Eq. (5)
        P_tilde_pseudoInverse = kron((P_training'/(P_training*P_training'))',eye(Nr)); %Compute pseudo-inverse of the effective training matrix
        
        average_MSE_MVU_estimator(k,statisticsIndex,1) = Nt^2*Nr/totalTrainingPower(k); %Compute the MSE using Eq. (12) in [8]
        
        H_hat = P_tilde_pseudoInverse'*(P_tilde*vecH_realizations + vecN_realizations); %Compute the estimate by Monte-Carlo simulations using Eq. (4) in [8]
        average_MSE_MVU_estimator(k,statisticsIndex,2) = mean( sum(abs(vecH_realizations - H_hat).^2,1) ); %Compute the MSE by Monte-Carlo simulations using the MSE definition
        
        
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
        
        average_MSE_onesided_estimator(k,statisticsIndex,1) = trace(inv(inv(R_T)+P_training*P_training'/Nr)); %Compute the MSE using Eq. (34) in [8]
        
        Ao = (P_training'*R_T*P_training + Nr*eye(Nt))\P_training'*R_T; %Compute the matrix Ao in the one-sided linear estimator using Eq. (25) in [8]
        H_hat = kron(transpose(Ao),eye(Nr))*(P_tilde*vecH_realizations + vecN_realizations); %Compute the estimate by Monte-Carlo simulations using Eq. (26) in [8]
        average_MSE_onesided_estimator(k,statisticsIndex,2) = mean( sum(abs(vecH_realizations - H_hat).^2,1) );  %Compute the MSE by Monte-Carlo simulations using the MSE definition
        
        
        %Two-sided linear estimator: Optimal training power allocation
        %
        %This heuristic estimator originates from the following paper:
        %
        %[27] D. Katselis, E. Kofidis, and S. Theodoridis, "On training optimization
        %for estimation of correlated MIMO channels in the presence of multiuser
        %interference," IEEE Trans. Signal Process., vol. 56, no. 10, pp.
        %4892-4904, 2008.
        
        P_training = diag(sqrt(q_uniform(:,k))); %Compute training matrix using Eq. (23) in [27], which becomes a uniform power allocation since we have uncolored noise in this simulation
        P_tilde = kron(P_training,eye(Nr)); %Compute effective training matrix as in Eq. (5)
        R_calE = sum(1./q_uniform(:,k))*eye(Nr); %Calculate the temporal covariance matrix of the processed interference term defined between Eq. (16) and Eq. (17) in [27].
        
        average_MSE_twosided_estimator(k,statisticsIndex,1) = trace(R_R-(R_R/(R_R+R_calE))*R_R); %Compute the MSE using Eq. (18) in [27]
        
        C1 = inv(P_training); %Compute C1 matrix according to Eq. (12) in [27]
        C2bar = R_R/(R_R+R_calE); %Compute C2bar matrix according to Eq. (17) in [27]
        H_hat = kron(transpose(C1),C2bar)*(P_tilde*vecH_realizations + vecN_realizations);
        average_MSE_twosided_estimator(k,statisticsIndex,2) = mean( sum(abs(vecH_realizations - H_hat).^2,1) ); %Compute the MSE by Monte-Carlo simulations using the MSE definition
        
    end
    
end


%Select a subset of training power for which we will plot markers
subset = linspace(1,length(totalTrainingPower_dB),5);

normalizationFactor = Nt*Nr; %Set MSE normalization factor to trace(R), so that the figures show normalized MSEs from 0 to 1.


%Plot the numerical results using the theoretical MSE formulas
figure(1); hold on; box on;

plot(totalTrainingPower_dB,mean(average_MSE_MVU_estimator(:,:,1),2)/normalizationFactor,'b:','LineWidth',2);

plot(totalTrainingPower_dB,mean(average_MSE_twosided_estimator(:,:,1),2)/normalizationFactor,'k-.','LineWidth',1);
plot(totalTrainingPower_dB,mean(average_MSE_onesided_estimator(:,:,1),2)/normalizationFactor,'r-','LineWidth',1);

plot(totalTrainingPower_dB(subset(1)),mean(average_MSE_MMSE_estimator_heuristic(subset(1),:,1),2)/normalizationFactor,'b+-.','LineWidth',1);
plot(totalTrainingPower_dB(subset(1)),mean(average_MSE_MMSE_estimator_optimal(subset(1),:,1),2)/normalizationFactor,'ko-','LineWidth',1);

legend('MVU, optimal','Two-sided linear, optimal','One-sided linear, optimal','MMSE, heuristic','MMSE, optimal','Location','SouthWest')

plot(totalTrainingPower_dB,mean(average_MSE_MMSE_estimator_heuristic(:,:,1),2)/normalizationFactor,'b-.','LineWidth',1);
plot(totalTrainingPower_dB,mean(average_MSE_MMSE_estimator_optimal(:,:,1),2)/normalizationFactor,'k-','LineWidth',1);
plot(totalTrainingPower_dB(subset),mean(average_MSE_MMSE_estimator_heuristic(subset,:,1),2)/normalizationFactor,'b+','LineWidth',1);
plot(totalTrainingPower_dB(subset),mean(average_MSE_MMSE_estimator_optimal(subset,:,1),2)/normalizationFactor,'ko','LineWidth',1);


set(gca,'YScale','Log'); %Set log-scale on vertical axis
xlabel('Total Training Power (dB)');
ylabel('Average Normalized MSE');
axis([0 totalTrainingPower_dB(end) 0.05 1]);

%Add title to Figure 1 to differentiate from Figure 2
title('Results based on theoretical formulas');


%Plot the numerical results using Monte-Carlo simulations
figure(2); hold on; box on;

plot(totalTrainingPower_dB,mean(average_MSE_MVU_estimator(:,:,2),2)/normalizationFactor,'b:','LineWidth',2);
plot(totalTrainingPower_dB,mean(average_MSE_twosided_estimator(:,:,2),2)/normalizationFactor,'k-.','LineWidth',1);
plot(totalTrainingPower_dB,mean(average_MSE_onesided_estimator(:,:,2),2)/normalizationFactor,'r-','LineWidth',1);
plot(totalTrainingPower_dB(subset(1)),mean(average_MSE_MMSE_estimator_heuristic(subset(1),:,2),2)/normalizationFactor,'b+-.','LineWidth',1);
plot(totalTrainingPower_dB(subset(1)),mean(average_MSE_MMSE_estimator_optimal(subset(1),:,2),2)/normalizationFactor,'ko-','LineWidth',1);

legend('MVU, optimal','Two-sided linear, optimal','One-sided linear, optimal','MMSE, heuristic','MMSE, optimal','Location','SouthWest')

plot(totalTrainingPower_dB,mean(average_MSE_MMSE_estimator_heuristic(:,:,2),2)/normalizationFactor,'b-.','LineWidth',1);
plot(totalTrainingPower_dB,mean(average_MSE_MMSE_estimator_optimal(:,:,2),2)/normalizationFactor,'k-','LineWidth',1);
plot(totalTrainingPower_dB(subset),mean(average_MSE_MMSE_estimator_heuristic(subset,:,2),2)/normalizationFactor,'b+','LineWidth',1);
plot(totalTrainingPower_dB(subset),mean(average_MSE_MMSE_estimator_optimal(subset,:,2),2)/normalizationFactor,'ko','LineWidth',1);


set(gca,'YScale','Log'); %Set log-scale on vertical axis
xlabel('Total Training Power (dB)');
ylabel('Average Normalized MSE');
axis([0 totalTrainingPower_dB(end) 0.05 1]);

%Add title to Figure 2 to differentiate from Figure 1
title('Results based on Monte-Carlo simulations');
