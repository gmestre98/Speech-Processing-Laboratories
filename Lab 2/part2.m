%ak = lpcauto(som,p,[intervalo , duracao, skip]) 
% p=16 good default value for Fs=16khz
%intervalo spacing between windows starts from the middle of the window
% skip of initial samples
% use overlapping winsows
[x, Fs] = audioread('birthdate_87005.wav');
%% Intervalo = 10ms / Duração = 20 ms / p= 16
intervalo = Fs*0.01;
duracao = Fs*0.02;
p = 16;
ak = lpcauto(x,p,[intervalo, duracao, 0]);

n_windows = length(ak);


[e, xr] = reconstruct (x,intervalo,ak, n_windows);

% audiowrite
str = 'test';
audiowrite([str,'birthdate_nnnnn_res.wav'],e,Fs);
audiowrite('birthdate_nnnnn_syn.wav',xr,Fs);

%% Intervalo = 10ms / Duração = 20 ms / p = 10


%% Intervalo = 5ms / Duração = 10 ms / p=10


%% Intervalo = 5ms / Duração = 10 ms / p=16





%% Intervalo