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
reconstruct (x,intervalo,ak, n_windows)



% audiowrite

