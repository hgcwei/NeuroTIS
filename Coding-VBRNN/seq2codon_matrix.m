function [x,y] = seq2codon_matrix(seq,cds,lws,rws)
len = length(seq);
y = zeros(1,len);
pos_inds = cds(1):3:cds(2);
y(1,pos_inds) = 1;
x = zeros(64,len);
s1 = blanks(lws);
s2 = blanks(rws);
s1(:) = 'N';
s2(:) = 'N';
seq_ext = [s1,seq,s2];
for i = 1:len
    x(:,i) = codon64(seq_ext(i:i+lws+rws))';
end
end

function [cnum] = codon64(seq)
[~,freqs] = codoncount(seq);
cnum = freqs(:)';
end