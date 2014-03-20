%This Matlab script can be used to generate Figure 3 in the article:
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


%Number of coupling matrices that the results are averaged over (it equals 200 in the paper)
nbrOfCouplingMatrices = 100;

Nt = 8; %Number of transmit antennas and length of the training sequence
Nr = 4; %Number of receive antennas

%Range of total training power in the simulation
totalTrainingPower_dB = 0:1:20; %In decibel scale
totalTrainingPower = 10.^(totalTrainingPower_dB/10); %In linear scale

%Scaling factors for the coupling matrices, where alpha^(j-1) is the
%scaling of the jth column.
alphaValues=[0.33 0.66 1];

%Options in the optimization Matlab algorithms used in this simulations
option = optimset('Display','off','TolFun',1e-7,'TolCon',1e-7,'Algorithm','interior-point');

%Minimum amount of power that is allocated in an eigendirection to count
%this eigendirection when computing the training length
closeToZero = 1e-3;


%Placeholders for storing the optimal and heuristic power allocation
powerallocationOptimal = zeros(Nt,length(totalTrainingPower),length(alphaValues),nbrOfCouplingMatrices);
powerallocationHeuristic = zeros(Nt,length(totalTrainingPower),length(alphaValues),nbrOfCouplingMatrices);

%Placeholders for storing the length of the training sequence with optimal
%and heuristic power allocation
trainingLengthOptimal = zeros(length(totalTrainingPower),nbrOfCouplingMatrices,length(alphaValues));
trainingLengthHeuristic = zeros(length(totalTrainingPower),nbrOfCouplingMatrices,length(alphaValues));

%Iteration over each realization of the random channel statistics
for statisticsIndex = 1:nbrOfCouplingMatrices
    
    %Generate coupling matrix V of the Weichselberger model with
    %chi-squared distributed variables (with 2 degrees of freedom).
    V = abs(randn(Nr,Nt)+1i*randn(Nr,Nt)).^2;
    
    %Iteration over each realization of the different scalings of the
    %column in the coupling matrix
    for alphaIndex = 1:length(alphaValues)
        
        %Compute the scaled coupling matrix using the current alpha value
        Vscaled = V*diag(alphaValues(alphaIndex).^(0:Nt-1));
        Vscaled = Nt*Nr*Vscaled/sum(Vscaled(:));
        
        %Compute the covariance matrix for the given coupling matrix
        R = diag(Vscaled(:));
        
        R_T = diag(sum(Vscaled,1)); %Compute transmit-side correlation matrix under the Weichselberger model
        
        %Compute the eigenvalues of the channel covariance matrices and
        %store them in decreasing order.
        eigenvaluesTransmitter = sort(eig(R_T),'descend');
        
        %Equal power allocation is initial point in optimization
        trainingpower_initial = ones(Nt,1)/Nt;
        
        
        %Compute optimal and heuristic power allocation for each total
        %training power.
        for k = 1:length(totalTrainingPower)
            
            %Compute optimal training power allocation for the MMSE estimator using built-in optimization algorithms
            powerallocationOptimal(:,k,alphaIndex,statisticsIndex) = fmincon(@(q) functionMSEmatrix(R,q,Nr),totalTrainingPower(k)*trainingpower_initial,ones(1,Nt),totalTrainingPower(k),[],[],zeros(Nt,1),totalTrainingPower(k)*ones(Nt,1),[],option);
            
            %Compute the optimal training length by checking how many
            %training powers that are larger than closeToZero
            trainingLengthOptimal(k,statisticsIndex,alphaIndex) = sum(powerallocationOptimal(:,k,alphaIndex,statisticsIndex)>closeToZero);
            
            %Update the initial power allocation as a preperation for the
            %next iteration
            trainingpower_initial = powerallocationOptimal(:,k,alphaIndex,statisticsIndex)/sum(powerallocationOptimal(:,k,alphaIndex,statisticsIndex));
            
            
            %Compute heuristic training power allocating according to Heuristic 1
            %in the paper, which uses Corollary 1 and Eq. (14) heuristically
            alpha_candidates = (totalTrainingPower(k)+cumsum(1./eigenvaluesTransmitter))./(1:Nt)'; %Compute different values on the Lagrange multiplier in Eq. (14), given that 1,2,...,Nt of the eigendirections get non-zero power
            optimalIndex = find(alpha_candidates-1./eigenvaluesTransmitter>0 & alpha_candidates-[1./eigenvaluesTransmitter(2:end); Inf]<0); %Find the true Lagrange multiplier alpha by checking which one of the candidates that only turns on the eigendirections that are supposed to be on
            powerallocationHeuristic(:,k,alphaIndex,statisticsIndex) = max([alpha_candidates(optimalIndex)-1./eigenvaluesTransmitter zeros(Nt,1)],[],2); %Compute the power allocation according to  Eq. (14) using the optimal alpha
            
            %Compute the optimal training length by checking how many
            %training powers that are larger than closeToZero
            trainingLengthHeuristic(k,statisticsIndex,alphaIndex) = sum(powerallocationHeuristic(:,k,alphaIndex,statisticsIndex)>closeToZero);
        end
        
    end
    
end


%Plot the numerical results
figure; hold on; box on;

plot(totalTrainingPower_dB(1),mean(trainingLengthHeuristic(1,:,1),2),'r-.');
plot(totalTrainingPower_dB(1),mean(trainingLengthOptimal(1,:,1),2),'k-');

legend('Heuristic Training Sequence','Optimal Training Sequence','Location','SouthEast')

plot(totalTrainingPower_dB,mean(trainingLengthOptimal(:,:,1),2),'ko-');
plot(totalTrainingPower_dB,mean(trainingLengthHeuristic(:,:,1),2),'ro-.');

plot(totalTrainingPower_dB,mean(trainingLengthOptimal(:,:,2),2),'k*-');
plot(totalTrainingPower_dB,mean(trainingLengthHeuristic(:,:,2),2),'r*-');

plot(totalTrainingPower_dB,mean(trainingLengthOptimal(:,:,3),2),'kd-');
plot(totalTrainingPower_dB,mean(trainingLengthHeuristic(:,:,3),2),'rd-');

%Write out the alpha-values as text
text(1,mean([trainingLengthHeuristic(3,:,1) trainingLengthOptimal(3,:,1)],2),['\alpha=' num2str(alphaValues(1))]);
text(1,mean([trainingLengthHeuristic(3,:,2) trainingLengthOptimal(3,:,2)],2),['\alpha=' num2str(alphaValues(2))]);
text(1,mean([trainingLengthHeuristic(3,:,3) trainingLengthOptimal(3,:,3)],2),['\alpha=' num2str(alphaValues(3))]);

xlabel('Total Training Power (dB)');
ylabel('Average Optimal Training Length');
