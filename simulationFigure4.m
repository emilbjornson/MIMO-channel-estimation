%This Matlab script can be used to generate Figure 4 in the article:
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

Nt = 8; %Number of transmit antennas and length of the training sequence
Nr = 4; %Number of receive antennas

%Range of total training power in the simulation
totalTrainingPower_dB = -5:1:20; %In decibel scale
totalTrainingPower = 10.^(totalTrainingPower_dB/10); %In linear scale

%Options in the optimization Matlab algorithms used in this simulations
option = optimset('Display','off','TolX',1e-7,'TolFun',1e-7,'TolCon',1e-7,'Algorithm','interior-point');

%Define the correlation between adjacent antennas in the exponential
%correlation model from [30]. Note that these parameters are generally
%complex-valued, where the absolute value determines the eigenvalues and
%the phase determines the corresponding eigenvectors. Since this simulation
%considers a point-to-point system without interference, we can ignore the
%phase without loss of generality.
antennaCorrelationTransmitter = 0.8;
antennaCorrelationReceiver = 0;

%Generate channel covariance matrices at the transmitter and receiver based
%on Kronecker model and the exponential correlation model from [30].
Rtx = toeplitz(antennaCorrelationTransmitter.^(0:Nt-1));
Rrx = toeplitz(antennaCorrelationReceiver.^(0:Nr-1));

%Compute the eigenvalues of the channel covariance matrices and store them
%in decreasing order.
eigenvaluesTransmitter = sort(eig(Rtx),'descend');
eigenvaluesReceiver = sort(eig(Rrx),'descend');

%Compute the channel covariance matrix. The covariance matrix has been
%diagonalized, without loss of generality since we consider a
%point-to-point system without interference.
R = kron(diag(eigenvaluesTransmitter),diag(eigenvaluesReceiver));

R_T = diag(eigenvaluesTransmitter)*sum(eigenvaluesReceiver); %Compute diagonalized transmit-side correlation matrix

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
[eigenvaluesTransmitter_sorted,permutationorder] = sort(eigenvaluesTransmitter,'descend'); %Compute and sort the eigenvalues
[~,inversePermutation] = sort(permutationorder); %Keep track of the order of eigenvalues

q_MMSE_heuristic = zeros(Nt,length(totalTrainingPower));
for k = 1:length(totalTrainingPower) %Go through each total training power
    alpha_candidates = (totalTrainingPower(k)+cumsum(1./eigenvaluesTransmitter_sorted))./(1:Nt)'; %Compute different values on the Lagrange multiplier in Eq. (14), given that 1,2,...,Nt of the eigendirections get non-zero power
    optimalIndex = find(alpha_candidates-1./eigenvaluesTransmitter_sorted>0 & alpha_candidates-[1./eigenvaluesTransmitter_sorted(2:end); Inf]<0); %Find the true Lagrange multiplier alpha by checking which one of the candidates that only turns on the eigendirections that are supposed to be on
    q_MMSE_heuristic(:,k) = max([alpha_candidates(optimalIndex)-1./eigenvaluesTransmitter_sorted zeros(Nt,1)],[],2); %Compute the power allocation according to  Eq. (14) using the optimal alpha
end

q_MMSE_heuristic = q_MMSE_heuristic(inversePermutation,:); %Finalize the heuristic power allocation by reverting the sorting of the eigenvalues


%Initialization of Monte-Carlo simulations
vecH_realizations = sqrt(R)*( randn(Nt*Nr,nbrOfMonteCarloRealizations) + 1i*randn(Nt*Nr,nbrOfMonteCarloRealizations) ) / sqrt(2); %Generate channel realizations on vectorized form. Each column is an independent realization of vec(H).
vecN_realizations = ( randn(Nt*Nr,nbrOfMonteCarloRealizations) + 1i*randn(Nt*Nr,nbrOfMonteCarloRealizations) ) / sqrt(2); %Generate noise realizations on vectorized form. Each column is an independent realization of vec(N).


%The maximal values of the Lagrange multiplier alpha>=0 for which a
%certain channel eigendirection (except the first one) is allocated
%non-zero training power. Larger eigenvalue means that larger alpha can
%be tolerated. The formula is found in Corollary 3. Note that the
%eigenvalues of the receive covariance matrix has been normalized to 1.
alphaLimits = 8*eigenvaluesTransmitter(2:end).^3/27;

%Generate an upper triangle matrix that mask out the active training
%powers when different number of eigendirections receive non-zero
%training power.
upperTriangle = toeplitz([1; zeros(Nt-1,1)],ones(Nt-1,1));

%Compute the training power coefficients in Eq. (21) for the limiting
%alpha-values from above. This gives the power that is allocated to
%each eigendirection at the value of the Lagrange multiplier when we
%are just about to active (allocate non-zero power) to one more
%eigendirection.
powerLimitsUnmasked = sqrt(8*eigenvaluesTransmitter*(1./alphaLimits)'/3).*cos(pi/3-atan(sqrt(8*eigenvaluesTransmitter.^3*(1./alphaLimits)'/27-1)))-repmat(1./eigenvaluesTransmitter,[1 length(alphaLimits)]);
powerLimits = powerLimitsUnmasked.*upperTriangle;

%Compute the maximal amount of training power, totalPowerLimits(k),
%that can be allocated to the first k eigendirections, without having
%to active eigendirection k+1.
totalPowerLimits = sum(powerLimits.*upperTriangle,1);

%Add zero to alphaLimits since the Lagrange multiplier needs to be
%non-negative.
alphaLimits = [alphaLimits; 0];


%Placeholder for the optimal training power allocation for different
%amount of total training power.
q_norm_MMSE = zeros(Nt,length(totalTrainingPower));

%Go through all total training powers and compute the optimal training
%power allocation by using Corollary 3.
for k = 1:length(totalTrainingPower)
    
    %If the total training power is smaller than the value when the
    %second eigendirection is activated, then all power is placed in
    %the dominating eigendirection.
    if  totalTrainingPower(k) <= totalPowerLimits(1)
        q_norm_MMSE(1,k) = totalTrainingPower(k);
        
    else
        
        %Use the maximal amount of training power for each number of
        %active eigendirections to determine how many eigendirections
        %that should be considered.
        difference = totalPowerLimits-totalTrainingPower(k);
        numberOfActive = sum(difference<0)+1;
        
        %Upper bound on the Lagrange multiplier, based on how many
        %eigendirections that need to be active.
        alphaMax = alphaLimits(numberOfActive-1);
        
        
        %Initialization in training power computation
        q_candidate = [totalTrainingPower(k); zeros(Nt-1,1)]; %Put all power in dominating eigendirection
        currentMinimumMSE = functionMSEnorm(eigenvaluesTransmitter,eigenvaluesReceiver,q_candidate); %Compute the corresponding MSE
        
        
        for j = 2:numberOfActive
            
            %Generate different combinations of the parameter k, which
            %takes the values 0 and 1, that appears in Eq. (21).
            bitstreams = dec2bin(0:2^j-1)-48;
            
            for l = 1:size(bitstreams,1)
                
                %Compute one power allocation from Corollary 3 by
                %finding the Langrange multiplier that gives equality
                %in the training power constraint.
                alphaOptimized = fminbnd(@(x) functionLagrangeMultiplier(eigenvaluesTransmitter(1:j),totalTrainingPower(k),bitstreams(l,:),x),0,alphaMax,option);
                [~,powerAllocation] = functionLagrangeMultiplier(eigenvaluesTransmitter(1:j),totalTrainingPower(k),bitstreams(l,:),alphaOptimized);
                
                q_new = [powerAllocation'; zeros(Nt-j,1)]; %This power allocation is padded with zeros to get an Nt x 1 vector.
                
                %Check if the power allocation makes sense, in the
                %sense that it uses most of the available training
                %power and is more-or-less positive and real-valued. If
                %this is the case, then the performance of the power
                %allocation is compared with the current candidate.
                if abs(sum(q_new)-totalTrainingPower(k))/totalTrainingPower(k)<0.01 && max(abs(imag(powerAllocation)))<1e-6 && min(powerAllocation)>-1e-3
                    
                    q_new = totalTrainingPower(k)*q_new/sum(q_new); %Make sure that the power allocation uses exactly all the available training power
                    newMSE = functionMSEnorm(eigenvaluesTransmitter,eigenvaluesReceiver,q_new); %Compute the corresponding MSE
                    
                    %Update the candidate power allocation if the new
                    %solution is better than the previous ones.
                    if currentMinimumMSE > newMSE
                        currentMinimumMSE = newMSE;
                        q_candidate = q_new;
                    end
                end
            end
        end
        
        q_norm_MMSE(:,k) = q_candidate; %Store the optimal training power allocation
    end
    
end


%Placeholders for storing the MSEs with different channel estimators.
%Second dimension is 1 for the theoretical MSE results and 2 for empirical 
%results using Monte-Carlo simulations
average_MSE_channelbased_uniform = zeros(length(totalTrainingPower),1);
average_MSE_channelbased_optimal = zeros(length(totalTrainingPower),1);
average_MSE_MMSE_uniform = zeros(length(totalTrainingPower),2);
average_MSE_MMSE_channeloptimized = zeros(length(totalTrainingPower),2);
average_MSE_MMSE_optimal = zeros(length(totalTrainingPower),2);


for k = 1:length(totalTrainingPower)
    
    %Uniform training power allocation
    Ptilde = kron(sqrt(totalTrainingPower(k))*eye(Nt)/sqrt(Nt),eye(Nr));
    
    %Estimate the full channel matrix and compute the squared norm of this estimate.
    Hhat = (R*Ptilde')/(Ptilde*R*Ptilde'+eye(Nt*Nr)) * (Ptilde*vecH_realizations + vecN_realizations);
    average_MSE_channelbased_uniform(k) = mean( abs(sum(abs(vecH_realizations).^2,1) - sum(abs(Hhat).^2,1)).^2 );
    
    
    %Optimal training power allocation for channel matrix
    %estimation (i.e., suboptimal for squared norm estimation)
    Ptilde = kron(diag(sqrt(q_MMSE_heuristic(:,k))),eye(Nr));
    
    %Estimate the full channel matrix and compute the squared norm of this estimate.
    Hhat = (R*Ptilde')/(Ptilde*R*Ptilde'+eye(Nt*Nr)) * (Ptilde*vecH_realizations + vecN_realizations);
    average_MSE_channelbased_optimal(k) = mean( abs(sum(abs(vecH_realizations).^2,1) - sum(abs(Hhat).^2,1)).^2 );
    
    
    %Uniform training power allocation
    Ptilde = kron(sqrt(totalTrainingPower(k))*eye(Nt)/sqrt(Nt),eye(Nr));
    B = R/(Ptilde*R*Ptilde'+eye(Nt*Nr)); %Compute matrix B as in [Theorem 3, 16]
    
    %Compute MSE theoretically (using matrix notation from [Theorem 3, 16])
    average_MSE_MMSE_uniform(k,1)=sum(sum(B*(2*Ptilde*R*Ptilde'+eye(Nt*Nr))*B));
    
    %Compute MSE by Monte-Carlo simualtions (using matrix notation from [Theorem 3, 16])
    normEstimates = sum(sum(B))+sum(abs(B*Ptilde*(Ptilde*vecH_realizations+vecN_realizations)).^2,1);
    average_MSE_MMSE_uniform(k,2)=mean( abs( sum(abs(vecH_realizations).^2,1) - normEstimates ).^2 );
    
    
    %Optimal training power allocation for channel matrix
    %estimation (i.e., suboptimal for squared norm estimation)
    Ptilde = kron(diag(sqrt(q_MMSE_heuristic(:,k))),eye(Nr));
    B = R/(Ptilde*R*Ptilde'+eye(Nt*Nr)); %Compute matrix B as in [Theorem 3, 16]
    
    %Compute MSE theoretically (using matrix notation from [Theorem 3, 16])
    average_MSE_MMSE_channeloptimized(k,1)=sum(sum(B*(2*Ptilde*R*Ptilde'+eye(Nt*Nr))*B));
    
    %Compute MSE by Monte-Carlo simualtions (using matrix notation from [Theorem 3, 16])
    normEstimates = sum(sum(B))+sum(abs(B*Ptilde*(Ptilde*vecH_realizations+vecN_realizations)).^2,1);
    average_MSE_MMSE_channeloptimized(k,2)=mean( abs( sum(abs(vecH_realizations).^2,1) - normEstimates ).^2 );
    
    
    %Optimal training power allocation for squared channel norm estimation
    Ptilde = kron(diag(sqrt(q_norm_MMSE(:,k))),eye(Nr));
    B = R/(Ptilde*R*Ptilde'+eye(Nt*Nr)); %Compute matrix B as in [Theorem 3, 16]
    
    %Compute MSE theoretically (using matrix notation from [Theorem 3, 16])
    average_MSE_MMSE_optimal(k,1) = sum(sum(B*(2*Ptilde*R*Ptilde'+eye(Nt*Nr))*B));
    
    %Compute MSE by Monte-Carlo simualtions (using matrix notation from [Theorem 3, 16])
    normEstimates = sum(sum(B)) + sum(abs(B*Ptilde*(Ptilde*vecH_realizations + vecN_realizations)).^2,1);
    average_MSE_MMSE_optimal(k,2) = mean( abs( sum(abs(vecH_realizations).^2,1) - normEstimates ).^2 );
    
end


%Select a subset of training power for which we will plot markers
subset = linspace(1,length(totalTrainingPower_dB),6);
subset = subset(2:end-1);

normalizationFactor = trace(R*R); %Set MSE normalization factor to trace(R)^2, so that the figures show normalized MSEs from 0 to 1.


%Plot the numerical results using the theoretical MSE formulas 
%(the channel-based case are computed by Monte-Carlo simulations)
figure(1); hold on; box on;

plot(totalTrainingPower_dB,average_MSE_channelbased_uniform/normalizationFactor,'r--','LineWidth',1);
plot(totalTrainingPower_dB,average_MSE_channelbased_optimal/normalizationFactor,'r','LineWidth',1);

plot(totalTrainingPower_dB(subset(1)),average_MSE_MMSE_uniform(subset(1),1)/normalizationFactor,'kd--','LineWidth',1);
plot(totalTrainingPower_dB(subset(1)),average_MSE_MMSE_channeloptimized(subset(1),1)/normalizationFactor,'k+-.','LineWidth',1);
plot(totalTrainingPower_dB(subset(1)),average_MSE_MMSE_optimal(subset(1),1)/normalizationFactor,'ko-','LineWidth',1);

legend('Channel-based, uniform','Channel-based, optimal','MMSE, uniform','MMSE, channel optimized','MMSE, optimal','Location','SouthWest')

plot(totalTrainingPower_dB,average_MSE_MMSE_uniform(:,1)/normalizationFactor,'k--','LineWidth',1);
plot(totalTrainingPower_dB,average_MSE_MMSE_channeloptimized(:,1)/normalizationFactor,'k-.','LineWidth',1);
plot(totalTrainingPower_dB,average_MSE_MMSE_optimal(:,1)/normalizationFactor,'k','LineWidth',1);

plot(totalTrainingPower_dB(subset),average_MSE_MMSE_uniform(subset,1)/normalizationFactor,'kd','LineWidth',1);
plot(totalTrainingPower_dB(subset),average_MSE_MMSE_channeloptimized(subset,1)/normalizationFactor,'k+','LineWidth',1);
plot(totalTrainingPower_dB(subset),average_MSE_MMSE_optimal(subset,1)/normalizationFactor,'ko','LineWidth',1);

set(gca,'YScale','Log')
xlabel('Total Training Power (dB)');
ylabel('Normalized MSE');

axis([-5 20 1e-2 10]);

%Add title to Figure 1 to differentiate from Figure 2
title('Results based on theoretical formulas');



%Plot the numerical results using Monte-Carlo simulations
figure(2); hold on; box on;

plot(totalTrainingPower_dB,average_MSE_channelbased_uniform/normalizationFactor,'r--','LineWidth',1);
plot(totalTrainingPower_dB,average_MSE_channelbased_optimal/normalizationFactor,'r','LineWidth',1);

plot(totalTrainingPower_dB(subset(1)),average_MSE_MMSE_uniform(subset(1),2)/normalizationFactor,'kd--','LineWidth',1);
plot(totalTrainingPower_dB(subset(1)),average_MSE_MMSE_channeloptimized(subset(1),2)/normalizationFactor,'k+-.','LineWidth',1);
plot(totalTrainingPower_dB(subset(1)),average_MSE_MMSE_optimal(subset(1),2)/normalizationFactor,'ko-','LineWidth',1);

legend('Channel-based, uniform','Channel-based, optimal','MMSE, uniform','MMSE, channel optimized','MMSE, optimal','Location','SouthWest')

plot(totalTrainingPower_dB,average_MSE_MMSE_uniform(:,2)/normalizationFactor,'k--','LineWidth',1);
plot(totalTrainingPower_dB,average_MSE_MMSE_channeloptimized(:,2)/normalizationFactor,'k-.','LineWidth',1);
plot(totalTrainingPower_dB,average_MSE_MMSE_optimal(:,2)/normalizationFactor,'k','LineWidth',1);

plot(totalTrainingPower_dB(subset),average_MSE_MMSE_uniform(subset,2)/normalizationFactor,'kd','LineWidth',1);
plot(totalTrainingPower_dB(subset),average_MSE_MMSE_channeloptimized(subset,2)/normalizationFactor,'k+','LineWidth',1);
plot(totalTrainingPower_dB(subset),average_MSE_MMSE_optimal(subset,2)/normalizationFactor,'ko','LineWidth',1);

set(gca,'YScale','Log')
xlabel('Total Training Power (dB)');
ylabel('Normalized MSE');

axis([-5 20 1e-2 10]);

%Add title to Figure 2 to differentiate from Figure 1
title('Results based on Monte-Carlo simulations');
