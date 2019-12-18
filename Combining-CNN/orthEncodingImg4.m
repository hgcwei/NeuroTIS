function orthEncodingImg4(order,k,seqArr,seqInd,label,dic,rf1,rf2,global_feas)
if k == 1
    tp = 'c1';
elseif k == 2
    tp = 'c2';
elseif k == 3
    tp = 'c3';
elseif k == 4
    tp = 's1';
elseif k == 5
    tp = 's2';
elseif k == 6
    tp = 's3';
elseif k == 7
    tp = 's4';
end

ord = '00000';
si = num2str(order);
m = length(si);
ord(6-m:5) = si;


for i = 1:size(seqArr,1)

    seq = seqArr(i,:);
    sInd = seqInd(i);
    
    sin = '000000';
    sind = num2str(sInd);
    n = length(sind);
    sin(7-n:6) = sind;
    
    fc = [global_feas(i,:)';zeros(301-17,1)];
    
    mat1 = orthEncoding(seq);
%     rf = readFrame(seq,net);
%     rf2 = readFrame(seq2,net);
    path = strcat(dic,tp,'.seq',ord,'.ps',sin,'.lb',num2str(label),'.csv');
%     imwrite(mat2gray([rf1(i,:)',rf2(i,:)',mat1]),path);
    
    dlmwrite(path,[fc,rf1(i,:)',rf2(i,:)',mat1]);
end
end

% function [mat ] = orthEncoding2(seq)
% l = length(seq);
% mat = zeros(l,1);
% for i = 1:l
%     s = seq(i);
%     if strcmp(s,'A')
%         mat(i,:) = 0;
%     elseif strcmp(s,'C')
%         mat(i,:) = 1;
%     elseif strcmp(s,'G')
%         mat(i,:) = 2;
%     elseif strcmp(s,'T')
%         mat(i,:) = 3;
%     elseif strcmp(s,'N')
%         mat(i,:) = 4;
%     end
% end
% end

function [mat ] = orthEncoding(seq)
l = length(seq);
mat = zeros(l,4);
for i = 1:l
    s = seq(i);
    if strcmp(s,'A')
        mat(i,:) = [1,0,0,0];
    elseif strcmp(s,'C')
        mat(i,:) = [0,1,0,0];
    elseif strcmp(s,'T')
        mat(i,:) = [0,0,1,0];
    elseif strcmp(s,'G')
        mat(i,:) = [0,0,0,1];
    else
        mat(i,:) = [0,0,0,0];
    end
end
end

% function [rf] = readFrame(seq,net)
% l = length(seq);
% rf = zeros(l,1);
% freqs = zeros(64,l - 83);
% for i = 43:(l - 41)
%     [~,freq] = codoncount(seq(i-42:i+41));
%     freqs(:,i-42) = freq(:);
% end
% rf(43:l - 41) = sim(net,freqs);
% end


