function MSE = functionMSEnorm(eigenvaluesTransmitter,eigenvaluesReceiver,powerAllocation)
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
%eigenvaluesTransmitter = Nt x 1 vector with eigenvalues at the
%                         transmitter side
%eigenvaluesReceiver    = Nr x 1 vector with eigenvalues at the
%                         receiver side
%powerAllocation        = Nt x 1 vector with training power allocation
%
%OUTPUT:
%MSE               = Mean Squared Error for estimation of the squared norm

MSE = sum(sum(((eigenvaluesTransmitter*eigenvaluesReceiver').^2 + 2*(powerAllocation.*eigenvaluesTransmitter.^3)*(eigenvaluesReceiver').^3)./(1+(powerAllocation.*eigenvaluesTransmitter)*eigenvaluesReceiver').^2));

