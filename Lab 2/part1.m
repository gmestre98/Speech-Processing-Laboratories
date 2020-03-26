%% One window F0 Calculation 
[x, Fs] = audioread('vowels_87005.wav');

dt = 1/Fs;
t = 0:dt:(length(x)*dt)-dt;
figure;
plot(t,x);
xlabel('Time')
ylabel('Signal')

i = Fs*1.685;
n = Fs*1.380;
f0max= 400;

s = x(i:i+n-1);
figure
plot(s);
hamm = hamming(n);

y =  x(i:i+n-1).*hamm;

[r,lags] = xcorr(y);
figure
plot(lags, r);

% [peaks,locs,w,p ] =  findpeaks(r);
% m = max(r);

[amp,loc] = max(r((n+(Fs/f0max)):end));

samples =loc + Fs/f0max;

f0 = Fs/samples;

%% Generalized

[x, Fs] = audioread('birthdate_87005');

total_time = lenght(x);

space = 10*0.001/16000;
i = 1;

l = n-1 + space;

interval = n-1+space;
while i < length(x)
    
    y = x(i,i+n-1);

    % xcorr(y)
    % calcular pico
    % pico em F0;
    
    i=i+interval;
end

% media 
%

    
    
    
