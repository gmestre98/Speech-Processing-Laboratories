%% Speech Processing Course - Laboratory 2
% Authors:
% Gonçalo Mestre: 87005
% Ricardo Antão: 87107
% Part 3
%% Load Files
% load audiofile
[x, Fs] = audioread('birthdate_87005.wav'); 
% load ideal excitation from part 2
dummy=load('ideal_e.mat');
ideal_e = dummy.e;

%% Generate Excitation e

% Defining lengths for the window and the loop and other important
% variables
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
    frame = x((i-1)*intervalo + 1 : (i+1)*intervalo);
    autocorr = xcorr(frame);
    % Doing the same for the ideal excitation and obtaining it's energy
    h = ideal_e((i-1)*intervalo + 1 : (i+1)*intervalo); 
    o = xcorr(h);
    exc_energy = max(o);

    % Getting the energy of the signal and normalizing it
    m = max(autocorr);
    norm = autocorr/m;
    
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
    
    % If it's unvoiced
    if f0==0     
       e((i-1)*intervalo + 1 : (i+1)*intervalo,1) = randn(1, length(frame))*0.001;
       prev_zeros=0;
    % If it's voiced
    else
      
      frame_e = zeros(size(frame));
      
      imp_space = round(Fs/f0);
      
      start = imp_space-prev_zeros;
      
      if start<1
         frame_e(1:imp_space:end) =  0.1;
      else
         frame_e(start:imp_space:end) =  0.1;
      end
      f = xcorr(frame_e);
      base = max(f);
      
      gain=exc_energy/base;
      
      frame_e = frame_e.*(exc_energy);
      
      e((i-1)*intervalo + 1 : (i+1)*intervalo,1) = frame_e;
      %Find impulse times
      k=find(frame_e);
      %Find the ones before half of the window
      impulses = find(k<intervalo);
      % If there's none
      if isempty(impulses)
          prev_zeros=0;
      % If there's atleast one
      else
          % Calculate number of zeros already there before first impulse
          prev_zeros=intervalo-k(impulses(length(impulses)));
      end
    end
    
   
    
end
noise = randn(intervalo/2,1)*0.001;
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
    [frame,Zi] = filter(1,ak(j,:),segment,Zi);


    xr = vertcat(xr,frame);

  end
 
 audiowrite('birthdate_87005_voc.wav',xr,Fs);
