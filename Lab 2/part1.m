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

y =  x(i:i+n-1);

f0 =  calcf0(y,n,Fs,f0max);

%% Generalized

[x, Fs] = audioread('birthdate_87005.wav');

total_time = length(x);

intervalo = 0.01*Fs;
f0max=400;
nwindows = total_time/intervalo;
F0 = [];
yn=[];
F0avg = [];
i=1;
for i=1:(nwindows)
    
    init =(i-1)*intervalo + 1;
    final=i*intervalo;
    y = x(init:final);

    f0 =  calcf0(y,intervalo,Fs,f0max);

    yn=vertcat(yn,y);
   
    
    F0 = vertcat(F0,f0);
    
    if f0~=0
    F0avg = vertcat(F0avg,f0);
    end
    
end

f0res = mean(F0avg);

    
    
    
