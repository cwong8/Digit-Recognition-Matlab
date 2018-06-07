% cd 'C:/Users/Christopher/Desktop/MAT 167/Programming Project'

% Loading in the handwritten digit database
load USPS;

% Checking what files are in our database
whos;

% We find 4 files:
% test_labels is a 10 x 4649 matrix of real numbers (double)
% test_patterns is a 256 x 4649 matrix of real numbers (double)
% train_labels is a 10 x 4649 matrix of real numbers (double)
% train_patterns is a 256 x 4649 matrix of real numbers (double)

% The arrays train_patterns and test_patterns contain a raster scan of the 
% 16x16 gray level pixel intensities that have been normalized to lie
% within the range [-1,1].
% The arrays train_labels and test_labels contain the true information about
% the digit images.

% That is, if the j th handwritten digit image in train_patterns truly 
% represents the digit i , then the (i +1, j )th entry of train_labels is
% +1, and all the other entries of the j th column of train_labels are -1.

% Loop that will turn the first 16 columns of test_patterns into matrices,
% transpose those matrices, and then plot them on a 4x4 plotting grid.
figure;
for k = 1:16
  % Reshaping a 256-length column vector into a 16x16 matrix
  A = reshape(train_patterns(:, k), 16, 16);
  % Plotting on the kth plot of our 4x4 plotting grid
  subplot(4, 4, k);
  imagesc(A');
end % For k
  
% Computing the mean digits in the train_patterns and putting them in a 
% 256 x 10 matrix called train_aves (one column per digit).
% Then we plot the digits as before using imagesc and subplot.
% Preallocating train_aves
train_aves = zeros(256, 10);
figure;
for k = 1:10
  % Gathering all images in train_patterns corresponding to digit k-1
  B = train_patterns(:, train_labels(k,:)==1);
  % Since 'mean' averages over columns and we want to average by row, we 
  % specify DIM = 2 to average by row.
  train_aves(:, k) = mean(B, 2);
  % Reshaping a 256-length column vector into a 16x16 matrix
  C = reshape(train_aves(:, k), 16, 16);
  % Plotting on the kth plot of our 4x4 plotting grid
  subplot(2, 5, k);
  imagesc(C');
end % For k
  
% Creating a matrix test_classif of size 10x4649 containing the squared
% Euclidean distance between each image in the test_patterns and each mean 
% digit image in train_aves.

% Preallocating test_classif
test_classif = zeros(10, 4649);
for k = 1:10
  % Computes the squared Euclidean distances between all of the test digit
  % images and the kth mean digits of the training dataset
  test_classif(k, :) = sum((test_patterns-repmat(train_aves(:,k),[1 4649])).^2);
end % For k

% Computing the classification results by finding the position index of the
% minimum of each column of test_classif.
% Putting the results in a vector test_classif_res of size 1x4649.

% Preallocating test_classif_res
test_classif_res = zeros(1, 4649);
for j = 1:4649
  % Finding the position index giving the minimum of the jth column of
  % test_classif and storing it in ind
  [tmp, ind] = min(test_classif(:,j));
  % Storing the indices in our vector test_classif_res
  test_classif_res(1, j) = ind;
end % For j

% Preallocating test_confusion
test_confusion = zeros(10, 10);
% The rows of test_confusion are the actual digits and the columns are the
% predicted digits
for k = 1:10
  % We begin by looping 1 row at a time
  tmp = test_classif_res(test_labels(k,:)==1);
  for j = 1:10
    % Within each row, we find the corresponding column value by checking
    % how many predicted digits there are for each number (0 to 9).
    test_confusion(k, j) = sum(tmp == j);
  end % For j
end % For k

% Preallocating train_u
train_u = zeros(256, 17, 10);
% Computing the rank 17 SVD for each digit (0 to 9) and only storing the
% left singular vectors (the matrix U) in the matrix train_u
for k = 1:10
  [train_u(:,:,k),tmp,tmp2] = svds(train_patterns(:,train_labels(k,:)==1),17);
end % For k

% Preallocating test_svd17
test_svd17 = zeros(17, 4649, 10);
% Computing the expansion coefficients of each test digit with respect to
% the 17 singular vectors of each training digit image set
for k=1:10
  test_svd17(:,:,k) = train_u(:,:,k)'*test_patterns;
end

% Preallocating rank17approx
rank17approx = zeros(256, 4649, 10);
% Rank 17 approximation of test digits using the 17 left singular vectors
% of the kth digit training set
for k = 1:10
  rank17approx(:,:,k) = train_u(:,:,k)*test_svd17(:,:,k);
end

% Preallocating test_svd17_res
test_svd17_res = zeros(10, 4649);
% Computing the error between each original test digit image and its rank 
% 17 approximation using the kth digit images in the training data set.
for k = 1:4649
  % We go through column by column because each column contains the pattern
  % for a single digit
  for i = 1:10
    % We put one digit at a time through the 10 different bases, one for
    % each digit (0 to 9) and compute the residual between our test digit
    % and the rank 17 approximation obtained by SVD through the training
    % data set. It is important to note that the norm function is computing
    % VECTOR norms, not matrix norms.
    test_svd17_res(i,k) = norm(test_patterns(:,k) - rank17approx(:,k,i), 2);
  end % For i
end % For k

% Computing the classification results by finding the position index of the
% minimum of each column of test_svd17_res.
% Preallocating test_svd17classif_res
test_svd17classif_res = zeros(1, 4649);
for j = 1:4649
  % Finding the position index giving the minimum of the jth column of
  % test_svd17_res and storing it in ind
  [tmp, ind] = min(test_svd17_res(:,j));
  % Storing the indices in our vector test_svd17classif_res of size 1x4649
  test_svd17classif_res(1, j) = ind;
end % For j

% The rows of test_svd17_confusion are the actual digits and the columns 
% are the predicted digits
% Preallocating test_svd17_confusion
test_svd17_confusion = zeros(10, 10);
for k = 1:10
  % We begin by looping 1 row (or digit) at a time
  tmp = test_svd17classif_res(test_labels(k,:)==1);
  for j = 1:10
    % Within each row, we find the corresponding column value by checking
    % how many predicted digits there are for each number (0 to 9).
    test_svd17_confusion(k, j) = sum(tmp == j);
  end % For j
end % For k




%%% Extra code for analysis %%%
% Preallocation accuracy vector acc
acc = zeros(1, 10);
% Finding the percent accuracy of simplest algorithm
for k = 1:10
  % Accuracy is number correct / total number for each digit
  acc(1, k) = test_confusion(k, k) / sum(test_confusion(:, k));
end

% Preallocation accuracy vector acc_svd17
acc_svd17 = zeros(1, 10);
% Finding the percent accuracy of SVD (rank 17) algorithm
for k = 1:10
  % Accuracy is number correct / total number for each digit
  acc_svd17(1, k) = test_svd17_confusion(k, k) / sum(test_svd17_confusion(:, k));
end