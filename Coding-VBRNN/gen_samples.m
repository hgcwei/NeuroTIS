clc
clear
load test_seqs4900cell
load test_cds4900cell

load train_seqs15000cell
load train_cds15000cell

ratio = 0.1;
v = 84;
for i = 1:4900 
    i
    seq = test_seqs4900cell{i};
    cds = test_cds4900cell{i};
    len = length(seq);
    [inds_p,inds_n] = get_pos_neg_inds(len,cds,ratio);
    
    file = ['coding_csv4900/' num2str(i) '.csv'];
    codon_matrix = importdata(file);
    for j = 1:length(inds_p)
        sample = get_one_coding_sample(codon_matrix,inds_p(j),v);
        dlmwrite(['Test_coding4900_84mouse/1/seq' num2str(i) '.' num2str(j) '.csv'],sample);
%         imwrite(mat2gray(sample),['Test_coding4900/1/seq' num2str(i) '.' num2str(j) '.png']);
    end
    
    for j = 1:length(inds_n)
        sample = get_one_coding_sample(codon_matrix,inds_n(j),v);
        dlmwrite(['Test_coding4900_84mouse/0/seq' num2str(i) '.' num2str(j) '.csv'],sample);
%         imwrite(mat2gray(sample),['Test_coding4900/0/seq' num2str(i) '.' num2str(j) '.png']);
    end
end

for i = 1:15000 
    i
    seq = train_seqs15000cell{i};
    cds = train_cds15000cell{i};
    len = length(seq);
    [inds_p,inds_n] = get_pos_neg_inds(len,cds,ratio);
    
    file = ['coding_csv15000/' num2str(i) '.csv'];
    codon_matrix = importdata(file);
    for j = 1:length(inds_p)
        sample = get_one_coding_sample(codon_matrix,inds_p(j),v);
        dlmwrite(['Train_coding15000_84human/1/seq' num2str(i) '.' num2str(j) '.csv'],sample);
%         imwrite(mat2gray(sample),['Test_coding4900/1/seq' num2str(i) '.' num2str(j) '.png']);
    end
    
    for j = 1:length(inds_n)
        sample = get_one_coding_sample(codon_matrix,inds_n(j),v);
        dlmwrite(['Train_coding15000_84human/0/seq' num2str(i) '.' num2str(j) '.csv'],sample);
%         imwrite(mat2gray(sample),['Test_coding4900/0/seq' num2str(i) '.' num2str(j) '.png']);
    end
end


