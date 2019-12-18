clc
clear;
load 'train_seqs15000cell.mat'
load 'train_cds15000cell.mat'
l = length(train_seqs15000cell);

% CodonMatrix15000 = cell(l,1);

for i = 1:l
    i
    [x,y] = seq2codon_matrix(upper(train_seqs15000cell{i}),train_cds15000cell{i},42,41);
%     imwrite(mat2gray(x),[num2str(i) '.png']);
    dlmwrite(['coding_csv15000/' num2str(i) '.csv'],x);
end

% save CodonMatrix15000 CodonMatrix15000
