%% Speech Processing Course - Laboratory 2
% Authors:
% Gonçalo Mestre: 87005
% Ricardo Antão: 87107
% Part 1

close all;
clc;

%% One window F0 Calculation
% For one of the 9 European Portuguese oral vowels
[x1, Fs1] = audioread('vowels_87005.wav');

% Adult male voice boundaries defined with some
% error margin
f0min = 60;
f0max = 200;

% Defining the Sampling Period
dt = 1/Fs1;

% Defining the Time variable
t = 0:dt:(length(x1)*dt)-dt;

% % Plotting the audio signal
% figure;
% plot(t,x1);
% xlabel('Time')
% ylabel('Signal')

% Selecting a small window of the signal
% corresponding to the chosen oral vowel
i = Fs1*1.685;
n = Fs1*1.380;
s = x1(i:i+n-1);

% % Plotting the corresponding part
% figure
% plot(s);

% Finding F0% Signal correlation
r = xcorr(s);

f0vowel =  calcf0(r,n,Fs1,f0max,f0min);

%% Computing the average f0 for the birthdate speech signal
[x, Fs] = audioread('birthdate_87005.wav');

% Defining lengths for the window and the loop
total_time = length(x);
intervalo = 0.01*Fs;
tamanho = 0.02*Fs;
nwindows = total_time/intervalo;

%Defining vectors to store the data
F0 = [];
yn=[];
F0good = [];

%Threshold for the unvoiced sounds
threshold = 0.3;

% Main loop
for i=1:(nwindows-1)
    % Getting the signal and it's correlation
    y = x((i-1)*intervalo + 1 : (i+1)*intervalo);
    r = xcorr(y);

    % Getting the energy of the signal and normalizing it
    m = max(r);
    norm = r/m;
    
    % Removing unvoiced sounds
    for j=1:length(norm)
        if(norm(j) < threshold)
            norm(j) = 0;
        end
    end    
    
    % Computing f0
    [f0, m2] =  calcf0(norm,tamanho,Fs,f0max, f0min);
    
    % Eliminating the frequency for sounds with really low energy
    if (m <= 10^-2)
        f0 = 0;
    end
    % Making f0=0 where the max is an unvoiced sound
    if (m2 == 0)
        f0 = 0;
    end
    
    F0 = vertcat(F0,f0);
    if (f0~=0)
        F0good = vertcat(F0good, f0);
    end
end

f0res = mean(F0);
f0resgood = mean(F0good);
    
f = fopen('birthdate_87005.myf0', 'wt');
fprintf(f, '%f\n', F0);
fclose(f);

    
