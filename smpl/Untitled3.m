% m = importdata('G:\\Halloween\\X2\\JBC\\0800\\FitBody_56.txt');
% plot3(m(:,1), m(:,2), m(:,3), '.'); hold on, axis equal, cameratoolbar


% tpts = importdata('G:\\Halloween\\X2\\JBC\\0800\\tPoseLandmark_56.txt');
% opts = importdata('G:\\Halloween\\X2\\JBC\\0800\\PoseLandmark_56.txt');
% ppts = importdata('G:\\Halloween\\X2\\JBC\\0800\\pPoseLandmark_56.txt');

% close all, figure(1),
% drawCoCoSkeleton(opts, 'r');
% hold on, axis equal, cameratoolbar
% plot3(tpts(:,1), tpts(:,2), tpts(:,3), 'g*');
% plot3(ppts(:,1), ppts(:,2), ppts(:,3), 'b*');
%
% for ii = 1:18
%    str = sprintf('%d', ii-1);
%    text(tpts(ii, 1), tpts(ii, 2), tpts(ii, 3), str);
% end
%
% for ii = 1:14
%    str = sprintf('%d', ii-1);
%    text(ppts(ii, 1), ppts(ii, 2), ppts(ii, 3), str);
% end



% cam = importdata('C:/temp/calib.txt');
%
% fp = fopen('par.txt', 'w');
% for ii = 1:1965
%     if cam(ii, 6) < 1
%         fprintf(fp,  '%.4d.png 1.0 0.0 1.0 0.0 1.0 1.0 0.0 0.0 1.0 ',cam(ii, 1));
%         fprintf(fp, '1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0\n');
%         continue;
%     end
%
%     rt = cam(ii, 18:23); R = rodrigues(rt(1:3));
%     fprintf(fp,  '%.4d.png %f 0.0 %f 0.0 %f %f 0.0 0.0 1.0 ',cam(ii, 1), cam(ii, 6), cam(ii, 9), cam(ii, 7), cam(ii, 10));
%     fprintf(fp, '%.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f\n', R(1,1), R(1,2), R(1,3), R(2,1), R(2,2), R(2, 3), R(3, 1), R(3, 2), R(3,3), rt(4), rt(5), rt(6));
% end
% fclose(fp);
count = 0;
for cid = 0:14
    str = sprintf('G:/Halloween/X2/%d/Frame2Corpus.txt', cid);
    dat = importdata(str);
    for ii = 1:size(dat, 1)
        str1 = sprintf('G:/Halloween/X2/%d/%.4d.png', dat(ii, 1), dat(ii, 3));
        img = imread(str1);
        str2 = sprintf('G:/Halloween/X2/Corpus2/%.4d.jpg', count);
        imwrite(img, str2);
        count = count+1;
    end
end