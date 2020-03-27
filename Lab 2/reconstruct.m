function [e,xr] = reconstruct(x,intervalo,ak, n_windows)
%RECONSTRUCT Reconstruct signal and calculate residual through the LPC
%coefficients
%
%Input:
%   x -- signal to be reconstructed
%   intervalo -- space between center of windows
%   n_windows -- number of windows used to calculate aks
%   ak -- LPC coefficients
%
%Output;
%   e -- residual
%   xr -- reconstructed signal
%

e = [];
Zi=[];
xr = [];
    for j=1:(n_windows)

        init = (j*intervalo-intervalo/2)+1;
        if j==1 % if it's the first window
            init=1;
        end
        final = j*intervalo+intervalo/2;
        if j==n_windows % if it's the last window
            final=length(x);
        end
        
        segment = x(init:final);
        [y,Zi] = filter(ak(j,:),1,segment,Zi);

        % 2 options
        %e(init:final) = y;
        
        e = vertcat(e,y);
        
      

    end
%% Reconstruct x based on e and ak
 Zi=[];
 
      for j=1:(n_windows)

        init = (j*intervalo-intervalo/2)+1;
        if j==1 % if it's the first window
            init=1;
        end
        final = j*intervalo+intervalo/2;
        if j==n_windows % if it's the last window
            final=length(x);
        end
        
        segment = e(init:final);
        [y,Zi] = filter(1,ak(j,:),segment,Zi);

        % 2 options
        %e(init:final) = y;
        
        xr = vertcat(xr,y);
        
        % use filter to calculate e[n]
        % [y,Zi]=filter(B,A,X,Zi)
        % b is aks
        % A is 1
        % X is signal

     end


end