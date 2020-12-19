function drawCoCoSkeleton(pts, color)

plot3(pts(:,1), pts(:, 2), pts(:, 3), '*', 'Color', color), axis equal, cameratoolbar, hold on

id = [0, 1] + 1; line(pts(id, 1), pts(id,2), pts(id, 3), 'Color',color)
id = [1, 2] + 1; line(pts(id, 1), pts(id,2), pts(id, 3), 'Color',color)
id = [2, 3] + 1; line(pts(id, 1), pts(id,2), pts(id, 3), 'Color',color)
id = [3, 4] + 1; line(pts(id, 1), pts(id,2), pts(id, 3), 'Color',color)
id = [1, 5] + 1; line(pts(id, 1), pts(id,2), pts(id, 3), 'Color',color)
id = [5, 6] + 1; line(pts(id, 1), pts(id,2), pts(id, 3), 'Color',color)
id = [6, 7] + 1; line(pts(id, 1), pts(id,2), pts(id, 3), 'Color',color)
id = [1, 8] + 1; line(pts(id, 1), pts(id,2), pts(id, 3), 'Color',color)
id = [8, 9] + 1; line(pts(id, 1), pts(id,2), pts(id, 3), 'Color',color)
id = [9, 10] + 1; line(pts(id, 1), pts(id,2), pts(id, 3), 'Color',color)
id = [1, 11] + 1; line(pts(id, 1), pts(id,2), pts(id, 3), 'Color',color)
id = [11, 12] + 1; line(pts(id, 1), pts(id,2), pts(id, 3), 'Color',color)
id = [12, 13] + 1; line(pts(id, 1), pts(id,2), pts(id, 3), 'Color',color)
id = [0, 14] + 1; line(pts(id, 1), pts(id,2), pts(id, 3), 'Color',color)
id = [0, 15] + 1; line(pts(id, 1), pts(id,2), pts(id, 3), 'Color',color)
id = [14, 16] + 1; line(pts(id, 1), pts(id,2), pts(id, 3), 'Color',color)
id = [15, 17] + 1; line(pts(id, 1), pts(id,2), pts(id, 3), 'Color',color)

for ii = 1:18
   str = sprintf('%d', ii-1);
   text(pts(ii, 1), pts(ii, 2), pts(ii, 3), str);
end