function [deviation,powerAllocation]=functionLagrangeMultiplier(eigenvaluesTransmitter,totalPower,k,alpha)
%Compute the MSE for estimation of the squared Frobenius norm of the
%channel matrix for a given training power allocation. This is used in the paper:
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
%
%INPUT:
%eigenvaluesTransmitter = Vector with the active eigenvalues at the
%                         transmitter side
%totalPower             = Total power of the training sequence
%k                      = Vector with k parameter values as in Corollary 3
%alpha                  = Langrange multiplier value
%
%OUTPUT:
%deviation              = Difference between available power and used power
%powerAllocation        = Training power allocation from Corollary 3

%Compute power allocation based on Eq. (21) in Corollary 3
powerAllocation = sqrt(8*(1./alpha(:))*eigenvaluesTransmitter'/3).*cos(repmat((-1).^k*pi/3,[length(alpha) 1])-atan(sqrt(8*(1./alpha(:))*(eigenvaluesTransmitter.^3)'/27-1))/3)-repmat(1./eigenvaluesTransmitter',[length(alpha) 1]);

%Deviation between total available power and the power that is used
deviation = abs(totalPower-sum(powerAllocation,2));