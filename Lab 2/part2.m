%% Speech Processing Course - Laboratory 2
% Authors:
% Gonçalo Mestre: 87005
% Ricardo Antão: 87107
% Part 2

%% Intervalo = 10ms / Duração = 20 ms / p= 16
[x, Fs] = audioread('birthdate_87005.wav');% read sound file
% set interval and duration of window
intervalo = Fs*0.01;
duracao = Fs*0.02;
% set p
p = 16;
% calculate LPC coefficientes
ak = lpcauto(x,p,[intervalo, duracao, 0]);
%Calculate number of windows
n_windows = length(ak);
% Reconstruct signal
[e, xr] = reconstruct (x,intervalo,ak, n_windows);

% audiowrite
str = '10_20_16';
audiowrite([str,'birthdate_nnnnn_res.wav'],e,Fs);
audiowrite([str,'birthdate_nnnnn_syn.wav'],xr,Fs);


%% Intervalo = 10ms / Duração = 20 ms / p= 10
clear all

[x, Fs] = audioread('birthdate_87005.wav');% read sound file
% set interval and duration of window
intervalo = Fs*0.01;
duracao = Fs*0.02;
% set p
p = 10;
% calculate LPC coefficientes
ak = lpcauto(x,p,[intervalo, duracao, 0]);
%Calculate number of windows
n_windows = length(ak);
% Reconstruct signal
[e, xr] = reconstruct (x,intervalo,ak, n_windows);

% audiowrite
str = '10_20_10';
audiowrite([str,'birthdate_nnnnn_res.wav'],e,Fs);
audiowrite([str,'birthdate_nnnnn_syn.wav'],xr,Fs);


%% Intervalo = 5 ms / Duração = 20 ms / p = 10
clear all
[x, Fs] = audioread('birthdate_87005.wav');
intervalo = Fs*0.005;
duracao = Fs*0.02;
p = 10;
ak = lpcauto(x,p,[intervalo, duracao, 0]);

n_windows = length(ak);


[e, xr] = reconstruct (x,intervalo,ak, n_windows);

% audiowrite
str = '5_20_10';
audiowrite([str,'birthdate_nnnnn_res.wav'],e,Fs);
audiowrite([str,'birthdate_nnnnn_syn.wav'],xr,Fs);


%% Intervalo = 5 ms / Duração = 20 ms / p = 16
clear all
[x, Fs] = audioread('birthdate_87005.wav');
intervalo = Fs*0.005;
duracao = Fs*0.02;
p = 16;
ak = lpcauto(x,p,[intervalo, duracao, 0]);

n_windows = length(ak);


[e, xr] = reconstruct (x,intervalo,ak, n_windows);

% audiowrite
str = '5_20_16';
audiowrite([str,'birthdate_nnnnn_res.wav'],e,Fs);
audiowrite([str,'birthdate_nnnnn_syn.wav'],xr,Fs);

%% Intervalo = 5ms / Duração = 10 ms / p=10
clear all

[x, Fs] = audioread('birthdate_87005.wav');
intervalo = Fs*0.005; 
duracao = Fs*0.01;

p = 10;

ak = lpcauto(x,p,[intervalo, duracao, 0]);


n_windows = length(ak);

[e, xr] = reconstruct (x,intervalo,ak, n_windows);

% audiowrite
str = '5_10_10';
audiowrite([str,'birthdate_nnnnn_res.wav'],e,Fs);
audiowrite([str,'birthdate_nnnnn_syn.wav'],xr,Fs);

%% Intervalo = 5ms / Duração = 10 ms / p=16
clear all

[x, Fs] = audioread('birthdate_87005.wav');

intervalo = Fs*0.005;

duracao = Fs*0.01;

p = 16;

ak = lpcauto(x,p,[intervalo, duracao, 0]);

n_windows = length(ak);

[e, xr] = reconstruct (x,intervalo,ak, n_windows);

% audiowrite
str = '5_10_16';
audiowrite([str,'birthdate_nnnnn_res.wav'],e,Fs);
audiowrite([str,'birthdate_nnnnn_syn.wav'],xr,Fs);

%% Intervalo = 10ms / Duração = 10 ms / p=16
clear all

[x, Fs] = audioread('birthdate_87005.wav');

intervalo = Fs*0.01;

duracao = Fs*0.01;

p = 16;

ak = lpcauto(x,p,[intervalo, duracao, 0]);

n_windows = length(ak);

[e, xr] = reconstruct (x,intervalo,ak, n_windows);

% audiowrite
str = '10_10_16';
audiowrite([str,'birthdate_nnnnn_res.wav'],e,Fs);
audiowrite([str,'birthdate_nnnnn_syn.wav'],xr,Fs);

%% Intervalo = 10ms / Duração = 10 ms / p=10
clear all

[x, Fs] = audioread('birthdate_87005.wav');

intervalo = Fs*0.01;

duracao = Fs*0.01;

p = 10;

ak = lpcauto(x,p,[intervalo, duracao, 0]);

n_windows = length(ak);

[e, xr] = reconstruct (x,intervalo,ak, n_windows);

% audiowrite
str = '10_10_10';
audiowrite([str,'birthdate_nnnnn_res.wav'],e,Fs);
audiowrite([str,'birthdate_nnnnn_syn.wav'],xr,Fs);