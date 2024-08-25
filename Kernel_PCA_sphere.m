% This script illustrates Principle Component Analysis
% (PCA) with two classes of data located close near a 
% sphere. Mapping to a higher dimension as well four
% kernels are used.
[x,y,z] = sphere;
figure;
mesh(x,y,z, 'FaceAlpha',0.7,'EdgeColor','k')
hold on;
N = 100;

data1 = mvnrnd([0 0],[0.3 0; 0 0.3],N);
noise = mvnrnd([0 0 0], 0.001 * eye(3),N);
data1x = cos(data1(:,1)).*cos(data1(:,2)) + noise(:,1);
data1y = cos(data1(:,1)).*sin(data1(:,2)) + noise(:,2);
data1z = sin(data1(:,1)) + noise(:,3);
scatter3(data1x, data1y, data1z,20,'ko','filled')

data2 = mvnrnd([1.5 -1.5],[0.3 0; 0 0.1],N);
noise = mvnrnd([0 0 0], 0.001 * eye(3),N);
data2x = cos(data2(:,1)).*cos(data2(:,2)) + noise(:,1);
data2y = cos(data2(:,1)).*sin(data2(:,2)) + noise(:,2);
data2z = sin(data2(:,1)) + noise(:,3);
scatter3(data2x, data2y, data2z,20,'ro','filled')

labels = [zeros(N,1); ones(N,1)];
perm = randperm(numel(labels),numel(labels));
X = [data1x data1y data1z ; data2x data2y data2z];
X = X(perm,:);
labels = labels(perm,:);

% PCA in 4 dim with fourth component being the sum of squares of the other
% three components.
Y = [data1x data1y data1z data1x.^2+data1y.^2+data1z.^2;...
    data2x data2y data2z data2x.^2+data2y.^2+data2z.^2];
Y = Y(perm,:);
[coeff, score] = pca(Y,'NumComponents',2,...
        'Centered',false);
figure;
scatter(score(labels==0,1), score(labels==0,2),...
    20,'MarkerEdgeColor','k','MarkerFaceColor','k');
hold on;
scatter(score(labels==1,1), score(labels==1,2),...
    20,'MarkerEdgeColor','r','MarkerFaceColor','r');

% Quadratic kernel.
K = (X * X' + 0.1*ones(2*N,2*N)).^2;
K = K - ones(2*N,2*N)*K/(2*N) - K*ones(2*N,2*N)/(2*N) + ...
    ones(2*N,2*N)*K*ones(2*N,2*N)/(2*N)^2;
[V,D] = eig(K,'vector');
V(:,1) = 1/sqrt(D(1))*V(:,1);
V(:,2) = 1/sqrt(D(2))*V(:,2);
R = K*V(:,1:2);
figure;
scatter(R(labels==0,1), R(labels==0,2),...
    20,'MarkerEdgeColor','k','MarkerFaceColor','k');
hold on;
scatter(R(labels==1,1), R(labels==1,2),...
    20,'MarkerEdgeColor','r','MarkerFaceColor','r');

% Hyperbolic tangent (Sigmoid) kernel)
K = tanh(5*X * X' + 0.5*ones(2*N,2*N));
K = K - ones(2*N,2*N)*K/(2*N) - K*ones(2*N,2*N)/(2*N) + ...
    ones(2*N,2*N)*K*ones(2*N,2*N)/(2*N)^2;
[V,D] = eig(K,'vector');
V(:,1) = 1/sqrt(D(1))*V(:,1);
V(:,2) = 1/sqrt(D(2))*V(:,2);
R = K*V(:,1:2);
figure;
scatter(R(labels==0,1), R(labels==0,2),...
    20,'MarkerEdgeColor','k','MarkerFaceColor','k');
hold on;
scatter(R(labels==1,1), R(labels==1,2),...
    20,'MarkerEdgeColor','r','MarkerFaceColor','r');

% Gaussian kernel
scale = 0.3;
X2 = X*X';
d = diag(X2);
L = repmat(d',2*N,1) - 2*X2 + repmat(d,1,2*N);
K = exp(-L/(2*scale^2));
K = K - ones(2*N,2*N)*K/(2*N) - K*ones(2*N,2*N)/(2*N) + ...
    ones(2*N,2*N)*K*ones(2*N,2*N)/(2*N)^2;
[V,D] = eig(K,'vector');
V(:,1) = 1/sqrt(D(1))*V(:,1);
V(:,2) = 1/sqrt(D(2))*V(:,2);
R = K*V(:,1:2);
figure;
scatter(R(labels==0,1), R(labels==0,2),...
    20,'MarkerEdgeColor','k','MarkerFaceColor','k');
hold on;
scatter(R(labels==1,1), R(labels==1,2),...
    20,'MarkerEdgeColor','r','MarkerFaceColor','r');

% Thin plate spline
K = L .* log(sqrt(L)+ones(2*N,2*N));
K = K - ones(2*N,2*N)*K/(2*N) - K*ones(2*N,2*N)/(2*N) + ...
    ones(2*N,2*N)*K*ones(2*N,2*N)/(2*N)^2;
[V,D] = eig(K,'vector');
V(:,1) = 1/sqrt(-D(1))*V(:,1);
V(:,2) = 1/sqrt(-D(2))*V(:,2);
R = K*V(:,1:2);
figure;
scatter(R(labels==0,1), R(labels==0,2),...
    20,'MarkerEdgeColor','k','MarkerFaceColor','k');
hold on;
scatter(R(labels==1,1), R(labels==1,2),...
    20,'MarkerEdgeColor','r','MarkerFaceColor','r');