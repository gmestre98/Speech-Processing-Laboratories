function [f0, m] =  calcf0(r,n,Fs,f0max, f0min)
% Finding the maximum value of the correlation
[m,loc] = max(r(n+round(Fs/f0max):n+round(Fs/f0min)));

% Obtaining the fundamental period
f0 = Fs/(loc + round(Fs/f0max));
end 