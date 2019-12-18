function cscell = csv2cell(path,num)
cscell = cell(num,1);
for i = 1:num
    z = importdata([path num2str(i) '.csv']);
    cscell{i} = z(:,2)';
end
end

