%% Computing the average f0 for the birthdate speech signal
[x, Fs] = audioread('birthdate_87005.wav');

% Defining lengths for the window and the loop
total_time = length(x);
intervalo = 0.01*Fs;
tamanho = 0.02*Fs;
nwindows = total_time/intervalo - 1;
p=16;
f0min = 60;
f0max = 200;

%Defining vectors to store the data
ak = lpcauto(x,p,[intervalo, tamanho, 0]);
F0=[];
e = [];
%Threshold for the unvoiced sounds
threshold = 0.3;

% Main loop
for i=1:(nwindows-1)
    % Getting the signal and it's correlation
    y = x((i-1)*intervalo + 1 : (i+1)*intervalo);
    r = xcorr(y);

    % Getting the energy of the signal and normalizing it
    exc_energy = max(o);
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
    F0(i) = f0;
    if f0==0     
     % e((i-1)*intervalo + 1 : (i+1)*intervalo,1) = randn(1, length(y));
       e((i-1)*intervalo + 1 : (i+1)*intervalo,1) = randn(1, length(y))*0.01;
      
    else
      z = zeros(size(y));
      
      imp_space = round(Fs/f0);
      
     
      z(1:imp_space:end) =  0.1;
    
      
      e((i-1)*intervalo + 1 : (i+1)*intervalo,1) = z;
      %Count zeros
     
    end
    
   
    
end
noise = randn(intervalo/2,1)*0.01;
e = vertcat(noise,e);
e = vertcat(e,noise);



%% Generate signal based on e

Zi=[];
xr=[];

  for j=1:(nwindows)

    init = (j*intervalo-intervalo/2)+1;
    if j==1 % if it's the first window
        init=1;
    end
    final = j*intervalo+intervalo/2;
    if j==nwindows % if it's the last window
        final=length(x);
    end

    segment = e(init:final);
    [y,Zi] = filter(1,ak(j,:),segment,Zi);


    xr = vertcat(xr,y);

  end
 
 audiowrite('birthdate_87005_voc_robo.wav',xr,Fs);
