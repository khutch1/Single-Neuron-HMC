% Load dataset
load sample_data.dat

X = [ones(8,1) sample_data(:,1:2)];

t = sample_data(:,3);

% Plot data
figure(1); clf
plot(X(1:5,2), X(1:5:4), 'ks'); hold on
plot(X(6:8,2), X(6:8,3), 'k*')
xlim([1 10]); ylim([0 10]); axis square
xlabel('x1'); ylabel('x2')

% Define posterior distribution for W
alpha = 0.1
y = @(W) sigmf(W*X', [1 00]); %y(a), sigma funct. (output)
Objective_funct = @(W) -(t'*log(y(W)') + (1-t')*log(1-y(W))') + alpha*sum(W.^2, 2)'/2; %objective function
Liklehood_funct = @(W) exp(-G(W)); %likelihood function for single neuron

grad = @(W) -(t' - y(W))*X + alpha*W;

%%% Hamiltonian algorithm %%%
% Initialize values
lag = 160
burn_in = 30000

T = burn_in + 30*lag
Tau = 20;
epsilon = 0.16;	
W_stored_sub = zeros(Tau, 3);
W_stored = zeros(1, 3);
W_term = zeros(T+1, 3);
accepted = 0;
W = [0 0 0];
W_stored(1,:) = W;
W_term = W;
E = Objective_funct(W);
gradE = grad(W);

% Loop T times
for i = 1:T
	P = randn(size(W));
	H = P*P'/2 + E;
	W_new = W;
	gradE_new = gradE;
% Take Tau "leapfrog" steps
for j = 1:Tau
	P = P - epsilon*gradE_new/2;
	W_new = W_new + epsilon*P;
	gradE_new = grad(W_new);
	P = P - epsilon*gradE_new/2;
	W_stored_sub(j,:) = W_new;
end
% Update H
	E_new = Objective_funct(W_new);
	H_new = P*P'/2 + E_new;
	dH = H_new - H;
	% Decide whether to accept
	if dH < 0
		accept = 1;
	elseif rand() < exp(-dH)
		accept = 1;
	else
		accept = 0;
	end
	if accept
		W = W_new; E = E_new; gradE = gradE_new;
	end

	accepted = accepted + accept;
	W_stored = [W_stored; W_stored_sub];
	W_term(i+1,:) = W;
end


acceptance_rate = accepted/(T-1)

% Sum sampled output functions to find average neuron output
W_indep = W_stored(burn_in+lag:lag:T,:); %start at end of burn in plus lag length, take samples at lag length intervals, stop at or before Tth row
learned_y = @(x) zeros(1, length(x));

for i = 1:length(W_indep)
	W = W_indep(i,:);
	learned_y = @(x) [learned_y(x); sigmf(W*x', [1 0])];
end
learned_y = @(x) sum(learned_y(x))/length(W_indep);

%processing sample data for acf.matrix
Sample_mean = mean(W_stored);
vector = zeros(T, 1);
temp = [0, 0];
for j = 1:T
	temp = W_stored(j,:)-Sample_mean;
	vector(j) = temp*temp';
end

% Plots
figure(1); clf

% Sample autocorrelation
subplot(1,2,1)
acf(vector, lag);

% Predictive distribution
subplot(1, 2, 2)
plot(X(1:4,2), X(1:4,3), 'ks'); hold on
plot(X(5:8,2), X(5:8,3), 'k*')
xlim([0 10]); ylim([0 10]); axis square

title('Predictive Distribution'); xlabel('x1'); ylabel('x2')
hold on

x1 = linspace(0, 10);
x2 = x1;
[x1 x2] = meshgrid(x1, x2);

learned_y_cont = reshape(learned_y([ones(10000, 1) x1(:) x2(:)]), 100, 100);
contour(x1, x2, learned_y_cont, [0.12 0.27 0.73 0.88], '--k'); hold on
contour(x1, x2, learned_y_cont, [0.5 0.5], 'k') 