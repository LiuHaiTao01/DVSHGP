function C = diagAB(Anm,Bmn)
% C (an n x 1 vector) is diag(Anm*Bmn)

C = sum(Anm.*Bmn',2); % n x 1

end 