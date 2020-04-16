%% Speech Processing Course - Laboratory 2
% Authors:
% Gonçalo Mestre: 87005
% Ricardo Antão: 87107
% Part 4
%%
clear all
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
% Variable mode
F0=[140 170];
intensity = [0.1 0.7];

% Fixed mode
% F0 = 146.01;
% intensity = 0.3;


% formant_synt(vowel, F0, duration (s), intensity, mode)
formant_synt(4,F0,3, intensity,2);
