function MSE = functionMSEmatrix(R_diag,q_powerallocation,B)
%Compute the MSE for estimation of the channel matrix for a given training
%power allocation. This is used in the paper:
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
%R_diag            = Nt Nr x Nt Nr diagonal covariance matrix
%q_powerallocation = Nt x 1 vector with training power allocation
%B                 = Length of the training sequence.
%
%OUTPUT:
%MSE               = Mean Squared Error for estimation of the channel matrix

P_tilde = kron(diag(sqrt(q_powerallocation)),eye(B));

MSE = trace(R_diag - R_diag*(P_tilde'/(P_tilde*R_diag*P_tilde'+eye(length(R_diag))))*P_tilde*R_diag);
