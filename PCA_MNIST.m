% Load MNIST data set.
cd data
gunzip *.gz
filePrefix = 'data';
files = {   "train-images-idx3-ubyte",...
            "train-labels-idx1-ubyte",...
            "t10k-images-idx3-ubyte",...
            "t10k-labels-idx1-ubyte"  };
fid = fopen(fullfile(filePrefix, char(files{1})), 'r', 'b');
magicNum = fread(fid, 1, 'uint32');
numImgs  = fread(fid, 1, 'uint32');
numRows  = fread(fid, 1, 'uint32');
numCols  = fread(fid, 1, 'uint32');
rawImgDataTrain = uint8(fread(fid, numImgs * numRows * numCols, 'uint8'));
fclose(fid);
rawImgDataTrain = reshape(rawImgDataTrain, [numRows, numCols, numImgs]);
rawImgDataTrain = permute(rawImgDataTrain, [2,1,3]);
fid = fopen(fullfile(filePrefix, char(files{2})), 'r', 'b');
magicNum  = fread(fid, 1, 'uint32');
numLabels = fread(fid, 1, 'uint32');
labelsTrain = fread(fid, numLabels, 'uint8');
fclose(fid);
cd ..

k =1;
data = rawImgDataTrain(:,:,labelsTrain<=k);
labels = labelsTrain(labelsTrain<=k);
perm = randperm(numel(labels),25);
subset = data(:,:,perm);
figure;
montage(subset)
X = reshape(data, 28*28,numel(labels));
X = X';
X = double(X);

% Perform Principal Component Analysis (PCA)
% with two principal components.
[coeff, score] = pca(X,'NumComponents',2,...
        'Centered',false);
reconstructed = score*  coeff';
reconstructed = reconstructed';
reconstructed = reshape(reconstructed,28,28,numel(labels));
reconstructedsubset = reconstructed(:,:,perm);
figure;
montage(reconstructedsubset)
figure;
gscatter(score(:,1), score(:,2),labels,'rg');
    
% Perform Principal Component Analysis (PCA)
% with three principal components.
[coeff, score,~,~,explained] = pca(X,'NumComponents',3,...
        'Centered',false);
reconstructed = score*  coeff';
reconstructed = reconstructed';
reconstructed = reshape(reconstructed,28,28,numel(labels));
reconstructedsubset = reconstructed(:,:,perm);
figure;
montage(reconstructedsubset)
figure;
scatter3(score(labels==0,1), score(labels==0,2),score(labels==0,3),...
    5,'MarkerEdgeColor','r','MarkerFaceColor','r');
hold on;
scatter3(score(labels==1,1), score(labels==1,2),score(labels==1,3),...
    5,'MarkerEdgeColor','g','MarkerFaceColor','g');
%scatter3(score(labels==2,1), score(labels==2,2),score(labels==2,3),...
%    5,'MarkerEdgeColor','b','MarkerFaceColor','b');
%scatter3(score(labels==3,1), score(labels==3,2),score(labels==3,3),...
%    5,'MarkerEdgeColor','c','MarkerFaceColor','c');
legend('0','1');%,'2','3');
hold off;
