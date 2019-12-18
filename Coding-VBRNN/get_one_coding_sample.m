function sample = get_one_coding_sample(codon_matrix,pos,v)
%ONE_CODING_SAMPLES �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
l = size(codon_matrix,2);
fst = pos - v;
thd = pos + v;
if fst < 1
    fst = 1;
end
if thd > l
    thd = l;
end
sample = codon_matrix(:,[fst,pos,thd]);
end

