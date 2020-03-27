function [f0] =  calcf0(y,n,Fs,f0max)

[r,lags] = xcorr(y);

[amp,loc] = max(r(n+(Fs/f0max):end));

samples =loc + Fs/f0max;

f0 = Fs/samples;
if f0<60
    f0=0;
end



end 