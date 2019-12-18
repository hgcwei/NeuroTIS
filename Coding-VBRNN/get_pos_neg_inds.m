function [inds_p,inds_n] = get_pos_neg_inds(len,cds,ratio)
pos_inds = cds(1):3:cds(2);
neg_inds = setdiff(1:len,pos_inds);
lp = length(pos_inds);
ln = length(neg_inds);
num = ceil(min([lp,ln])*ratio);


rp = randperm(lp);
rn = randperm(ln);
inds_p = pos_inds(rp(1:num));
inds_n = neg_inds(rn(1:num));
end

