clc
clear
load train_seqs15000cell
load train_cds15000cell
load coding_scores15000cell

load test_seqs4900cell
load test_cds4900cell
load coding_scores4900cell


for i = 1:15000
    i
    k = 1;
        [pArr,pInd,nArr,nInd,psArr,nsArr,posRf,negRf,psRf,nsRf,p21feas,n21feas] = extractSglArr2(train_seqs15000cell{i},train_cds15000cell{i},150,150,k,coding_scores15000cell{i});
%         pathPos = strcat(trainSPath{k},'\1\');
%         pathNeg = strcat(trainSPath{k},'\0\');
%         orthEncodingImg2(i,4,upper(rna),pInd,1,'Train3\s1\1\',net);
        orthEncodingImg4(i,4,pArr,pInd,1,'Train15000_5\1\',posRf,psRf,p21feas);
%         orthEncodingImg3(i,4,pArr,pInd,1,'TestCE178_5_30\1\',netC1);
        
        ml = floor(min([size(pArr,1)*5,size(nArr,1)]));
        r = randperm(size(nArr,1));

%         ml2 = floor(min([size(pArr,1)*30,size(nArr,1)]));
%         r2 = randperm(size(nArr,1));
        
        orthEncodingImg4(i,4,nArr(r(1:ml),:),nInd(r(1:ml)),0,'Train15000_5\0\',negRf(r(1:ml),:),nsRf(r(1:ml),:),n21feas(r(1:ml),:));
%         orthEncodingImg3(i,4,nArr(r2(1:ml2),:),nInd(r2(1:ml2)),0,'TestCE178_5_30\0\',netC1);
%         orthEncodingImg2(i,4,nArr,nInd,0,'Test_Total_rf\s1\0\',net);

end



for i = 1:4900
    i
    k = 1;
        [pArr,pInd,nArr,nInd,psArr,nsArr,posRf,negRf,psRf,nsRf,p21feas,n21feas] = extractSglArr2(test_seqs4900cell{i},test_cds4900cell{i},150,150,k,coding_scores4900cell{i});
%         pathPos = strcat(trainSPath{k},'\1\');
%         pathNeg = strcat(trainSPath{k},'\0\');
%         orthEncodingImg2(i,4,upper(rna),pInd,1,'Train3\s1\1\',net);
        orthEncodingImg4(i,4,pArr,pInd,1,'Test4900_5\1\',posRf,psRf,p21feas);
%         orthEncodingImg3(i,4,pArr,pInd,1,'TestCE178_5_30\1\',netC1);
        
        ml = floor(min([size(pArr,1)*5,size(nArr,1)]));
        r = randperm(size(nArr,1));

%         ml2 = floor(min([size(pArr,1)*30,size(nArr,1)]));
%         r2 = randperm(size(nArr,1));
        
        orthEncodingImg4(i,4,nArr(r(1:ml),:),nInd(r(1:ml)),0,'Test4900_5\0\',negRf(r(1:ml),:),nsRf(r(1:ml),:),n21feas(r(1:ml),:));
%         orthEncodingImg3(i,4,nArr(r2(1:ml2),:),nInd(r2(1:ml2)),0,'TestCE178_5_30\0\',netC1);
%         orthEncodingImg2(i,4,nArr,nInd,0,'Test_Total_rf\s1\0\',net);

end







