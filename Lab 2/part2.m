%ak = lpcauto(som,p,[intervalo , duracao, skip]) 
% p=16 good default value for Fs=16khz
%intervalo spacing between windows starts from the middle of the window
% skip of initial samples
% use overlapping winsows
[x, Fs] = audioread('birthdate_87005.wav');

intervalo = Fs*0.01;
duracao = Fs*0.02;
p = 16;
ak = lpcauto(x,p,[intervalo, duracao, 0]);

n_windows = length(ak);

% 2 loops
for j=1:n_windows
init = (j-1)*intervalo+1;
final = j*intervalo;
x((j))
% use filter to calculate e[n] 
% filter(B,A,X)
% b is aks
% A is 1
% X is signal
    
end


% audiowrite