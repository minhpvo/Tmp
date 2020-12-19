warning off

dpath = 'G:/Halloween/X2';

ts = [0	0
1	5
2	-8
3	-4
4	3
5	-37
6	-1794
7	-2
8	-9
9	-1799
10	-12
11	9
12	-1749
13	-5];

 pidIn = [0]; cidIn = [3]; fidIn = 800;


for ii = 1:length(cidIn)
    cid = cidIn(ii); pid = pidIn(ii); fid = fidIn;
    
    str = sprintf('%s/%d/%.4d.png', dpath, cid, fid - ts(cid+1, 2));  img = imread(str);
    str = sprintf('%s/JBC/%.4d/%d.txt', dpath, fid, cid); prediction = importdata(str);
    

    plot_visible_limbs(img, prediction);
    
    hold on, zoom on
    str = sprintf('%d', fid); text(50, 50, str, 'fontsize', 20, 'color', 'g');
    for pid = 0:size(prediction, 1)/18-1
        for jid = 1:18
            if prediction(18*pid+jid, 3) >0
                str = sprintf('%d', pid); text(prediction(18*pid+jid, 1), prediction(18*pid+jid, 2), str, 'fontsize', 20, 'color', 'g');
                break;
            end
        end
    end
%                 hold off,
    
%     pts =  prediction(18*pid+[1:18],:);
%     figure(ii); plot_visible_limbs(img,pts);
%     id = find(pts(:,1) > 0); pts2 = pts(id, :);
%     minX = min(pts2(:, 1))-1920/60; maxX = max(pts2(:,1))+1920/60;
%     minY = min(pts2(:,2))-1920/60; maxY = max(pts2(:,2))+1920/60;
%     rectangle('Position', [minX, minY, maxX - minX, maxY - minY], 'EdgeColor','b', 'LineWidth', 3)
%     hold off
    %             thisimage = getframe; img2 = thisimage.cdata;
    %             str = sprintf('%d/%.4d', cid, fid); position =  [50 50];
    %             RGB = insertText(img2, position, str, 'FontSize', 18);
    %             writeVideo(writerObj, RGB);
end


