function y = formant_synt(vowel, F0, duration, intensity, mode)
    % vowel =
    % 1 -> i
    % 2 -> e
    % 3 -> E
    % 4 -> a
    % 5 -> 6
    % 6 -> @
    % 7 -> O
    % 8 -> o
    % 9 -> u

    % mode=
    % 1 -> fixed period
    % 2 -> variable frequency
    %% Initializing values and coefficients
    Fs= 8000;

    prev_zeros=0;

    % Duration of the signal in samples
    signal_duration = duration*Fs;

    % Sampling period
    Ts = 1/Fs;

    

    % Loading of formant table for vowels
    dummy = load('formant_table.mat');
    formant_table = dummy.formant_table;

    % choosing which formants according to vowel
    formants = formant_table(vowel,:);

    % calculating coefficients
    bandwidth=0.95;
    C = -bandwidth^2;
    B = 2*bandwidth*cos(2*pi*formants*Ts);
    A = ones(1,4)- B - C*ones(1,4);
%% Creating impulse train
    if mode == 1
        % period between impulses in samples
        imp_space = round(Fs/F0);
        % inicialization of excitation signal
        e = zeros(1,signal_duration);
        e(1:imp_space:end) =  intensity;
        
    elseif mode == 2
        % Initilization of needed parameters
        e=[];
        
        intervalo = Fs*0.1;
        nwindows = signal_duration/intervalo - 1;
        % Pseudo interpolation
        freq = linspace(F0(1), F0(2),nwindows-1);
        amp = linspace(intensity(1), intensity(2),nwindows-1);
        for i= 1:nwindows-1

            
%             % choosing which frequency for the frame      
%             if rand<=0.5;
%             new_f = f0min;
%             else
%             new_f = f0max;
%             end

            new_f = freq(i);
            % period in samples
            imp_space = round(Fs/new_f);
            frame = zeros(1,intervalo);
            % Calculating first impulse instant
            start = imp_space - prev_zeros;

           
            if start<1
                 % In case there were too many zeros before the start of
                 % the new frame
                frame(1:imp_space:end) = amp(i);
            else
                
                frame(start:imp_space:end) =  amp(i);
            end

            e = horzcat(e,frame);
            
            % Calculating number of zeros for transition between frames
            k=find(frame);
            
            if isempty(k)
                prev_zeros=0;
            else
                
                prev_zeros=intervalo-k(length(k));
            end
        end
        
    end
%% Generate output vowel


    % Concatenation of filters
    T1 = filter(A(1), [1 -B(1) -C], e);
    T2 = filter(A(2), [1 -B(2) -C], T1);
    T3 = filter(A(3), [1 -B(3) -C], T2);
    T4 = filter(A(4), [1 -B(4) -C], T3);

    silence = zeros(1, length(e)/2);
    y = horzcat(silence, T4);

    y = horzcat(y,silence);
    % added normalization to avoid clipping
    y = y/max(abs(y));

%% Write audio file
    if mode ==1
    audiowrite('formant_synthesis_fixed.wav', y, Fs);  
    elseif mode == 2
    audiowrite('formant_synthesis_var.wav', y, Fs);
    end



end



