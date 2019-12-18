function [posArr,posInd,negArr,negInd,psArr,nsArr,posRf,negRf,psRf,nsRf,pos21feas,neg21feas] = extractSglArr2(DNAseq,cdsInd,lws,rws,tp,featureMap)
% 提取一个序列中的信号，返回posArr,negArr
posArr = [];
posInd = [];
negArr = [];
negInd = [];
psArr = [];
nsArr = [];
posRf = [];
negRf = [];
psRf = [];
nsRf = [];
pos21feas = [];
neg21feas = [];
len = length(cdsInd);
if tp == 1
    siteInd = cdsInd(1:len:len);
elseif tp == 2
    siteInd = cdsInd(2:2:len-2);
elseif tp == 3
    siteInd = cdsInd(3:2:len);
else
    if ~isempty(cdsInd)
        siteInd = cdsInd(len);
    else
        siteInd = [];
    end
end
dnaLen = length(DNAseq);
s1 = blanks(lws);
s2 = blanks(rws);
s1(:) = 'N';
s2(:) = 'N';
dnaExt = [s1,DNAseq,s2];
rf = [zeros(1,lws),featureMap,zeros(1,rws)];
for i = lws + 1:lws + dnaLen -1 
    if tp == 1
        condition1 = strcmp(dnaExt(i:i+2),'ATG');
        condition2 = ~isempty(intersect(siteInd,i-lws));
    elseif tp == 2
        condition1 = strcmp(dnaExt(i:i+1),'GT');
        condition2 = ~isempty(intersect(siteInd,i-lws-1));
    elseif tp == 3
        condition1 = strcmp(dnaExt(i-2:i-1),'AG');
        condition2 = ~isempty(intersect(siteInd,i-lws));
    else
        condition1 = strcmp(dnaExt(i-3:i-1),'TAA') || strcmp(dnaExt(i-3:i-1),'TGA') || strcmp(dnaExt(i-3:i-1),'TAG');
        condition2 = ~isempty(intersect(siteInd,i-lws-1));
    end
    if condition1
        if condition2
            posArr = [posArr;dnaExt(i-lws:i+rws)];
            posInd = [posInd;i-lws];
            [sArr,sInd] = stopArrInd(dnaExt,i,lws,rws);
            psArr = [psArr;sArr];
            posRf = [posRf;rf(i-lws:i+rws)];
            
            if isempty(sInd)
                psRf = [psRf;zeros(lws+rws+1)];
                pos21feas = [pos21feas;zeros(1,17)];
            else
                psRf = [psRf;rf(sInd - lws:sInd+rws)];
                pos21feas = [pos21feas;global_fea(DNAseq,featureMap,i-lws,sInd-lws)];
            end
        else
            negArr = [negArr;dnaExt(i-lws:i+rws)];
            negInd = [negInd;i-lws];
            [sArr,sInd] = stopArrInd(dnaExt,i,lws,rws);
            nsArr = [nsArr;sArr];
            negRf = [negRf;rf(i-lws:i+rws)];
            
            if isempty(sInd)
                nsRf = [nsRf;zeros(lws+rws+1)];
                neg21feas = [neg21feas;zeros(1,17)];
            else
                nsRf = [nsRf;rf(sInd-lws:sInd+rws)];
                neg21feas = [neg21feas;global_fea(DNAseq,featureMap,i-lws,sInd-lws)];
            end
        end
    end
end
end

function [sArr,sInd] = stopArrInd(dnaExt,i,lws,rws)
sArr = blanks(lws+rws+1);
sInd = [];
for j = i+3:3:length(dnaExt)-rws-3
    if strcmp(dnaExt(j:j+2),'TAA') || strcmp(dnaExt(j:j+2),'TGA') || strcmp(dnaExt(j:j+2),'TAG')
        sArr = dnaExt(j-lws:j+rws);
        sInd = j;
        break;
    end
end
end

function [global_features] = global_fea(seq,coding_scores,i,j)
len = length(seq);
eta = 1e-100;
global_features = zeros(1,17);
% 1. the length of the upstream sequence to an ATG
global_features(1) = i-1;
% 2. the length of the downstream sequence to an ATG
global_features(2) = len-i-2;
% 3. the log ratio of (2) to (1)
global_features(3) = log((global_features(2)+eta)/(global_features(1)+ eta));
% 4. the number of upstream ATGs to an ATG
global_features(4) = length(strfind(seq(1:i-1),'ATG'));
% 5. the number of downstream ATGs to an ATG
global_features(5) = length(strfind(seq(i+3:len),'ATG'));
% 6. the log ratio of (5) to (4)
global_features(6) = log((global_features(5)+eta)/(global_features(4) + eta));
% 7. the number of in-frame upstream ATGs to an ATG
s1 = seq((i - floor((i-1)/3)*3):(i-1));
global_features(7) = inframeATG(s1);
% 8. the number of in-frame downstream ATGs to an ATG
s2 = seq(i+3:len);
global_features(8) = inframeATG(s2);
% 9. the log ratio of (8) to (7)
global_features(9) = log((global_features(8)+eta)/(global_features(7) + eta));
% 10. the number of upstream stop codon to an ATG
global_features(10) = length(strfind(seq(1:i-1),'TAA')) + length(strfind(seq(1:i-1),'TAG')) + length(strfind(seq(1:i-1),'TGA'));
% 11. the number of downstream stop codon to an ATG
global_features(11) = length(strfind(seq(i+3:len),'TAA')) + length(strfind(seq(i+3:len),'TAG')) + length(strfind(seq(i+3:len),'TGA'));
% 12. the log ratio of (11) to (10)
global_features(12) = log((global_features(11)+eta)/(global_features(10) + eta));
% 13. the number of in-frame upstream stop codons to an ATG
s1 = seq((i - floor((i-1)/3)*3):(i-1));
global_features(13) = inframeStop(s1);
% 14. the number of in-frame downstream stop codons to an ATG
s2 = seq(i+3:len);
global_features(14) = inframeStop(s2);
% 15. the log ratio of (8) to (7)
global_features(15) = log((global_features(14)+eta)/(global_features(13) + eta));
% 16. the length of the open reading frame starting at an ATG
global_features(16) = j-i+3;

% --------------- new features --------------------
% 17-19. the 1st, 2nd, 3rd frame mean value of open reading frame
% global_features(17) = mean(coding_scores(i:3:j));
% global_features(18) = mean(coding_scores(i+1:3:j+1));
% global_features(19) = mean(coding_scores(i+1:3:j+1));
% 
% % 20-21. the upstream and downstream mean value of coding scores
% global_features(20) = mean(coding_scores(j+4:len));
% global_features(21) = mean(coding_scores(1:i-1));
% if isnan(global_features(20))
%     global_features(20) = 0;
% end
% if isnan(global_features(21))
%     global_features(21) = 0;
% end
global_features(17) = global_coding(coding_scores,i,j);
end

function [num] = inframeATG(s)
count = codoncount(s);
num = count.ATG;
end

function [num] = inframeStop(s)
count = codoncount(s);
num1= count.TAA;
num2= count.TAG;
num3= count.TGA;
num = num1 + num2 + num3;
end

function [mc] = global_coding(scores,i,j)
y_ = zeros(1,length(scores));
y_(i:3:j) = 1;
mc = mean(abs(scores - 1 + y_));
end












