clc
% clear;
% load 'test_seqs4900cell.mat'
% load 'test_cds4900cell.mat'
% l = length(test_seqs4900cell);

% CodonMatrix4900 = cell(l,1);

for i = 1:4900
    i
%     [x,y] = seq2codon_matrix(upper(test_seqs4900cell{i}),test_cds4900cell{i},42,41);
%     CodonMatrix4900{i} = x;
%     imwrite(mat2gray(x),[num2str(i) '.png']);
dlmwrite(['coding_csv4900/' num2str(i) '.csv'],CodonMatrix4900{i});
end

% save CodonMatrix4900 CodonMatrix4900
