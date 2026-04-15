% clear;clc;
% path = "G:\";
% path_list = dir(path);
% for i = 3:length(path_list)
%     img_path = strcat(path,path_list(i).name);
%     img = load(img_path);
%     pred = img.pred;
%     label = img.label;
%     for j = 1:14
%         imwrite(pred(:,:,j),strcat("G:\a\","pred_",num2str(i),"channel",num2str(j),".png"), 'png');
%     end
% end

clear;clc;

img = load("G:\Work\论文\JSTSP2025\result\me\transRWKVb1dctv1\KAISTrealbest.mat");
input = img.input;
pred = img.pred;
truth = img.label;
%shape = size(pred);3
%pr = permute(pred,[4,5,3,1,2]); num,1,band,h,w
tr = permute(truth,[4,5,3,1,2]);

img = load("G:\Work\论文\SCI2024\转\extend_demosaic\PADUT_KAISTrealbest.mat");
% img = load("G:\Work\论文\JSTSP2025\result\fft\noise10\NTIRErealbest.mat");
pred = img.pred;
shape = size(pred);
pr = permute(pred,[4,5,3,1,2]);

                                                                                                                                                                                                
bands = 16;
h = shape(4);
w = shape(5);
pixel_num = h*w;
num = shape(1);
ratio_ergas = 1;
ss = zeros(16,num);
ps = zeros(16,num);
aux = zeros(16,num);
mean_y = zeros(16,num);


for j =1:16
    for i =1:num
        ss(j,i) = ssim(pr(:,:,j,i),tr(:,:,j,i));
        ps(j ,i) = psnr(pr(:,:,j,i),tr(:,:,j,i));
        
        % RMSE
        aux(j,i) = sum(sum((pr(:,:,j,i) - tr(:,:,j,i)).^2, 1), 2)/pixel_num;
        rmse_per_band = sqrt(aux);
        mean_y(j,i) = sum(sum(tr(:,:,j,i), 1), 2)/pixel_num;
%         fprintf("%.4f\n",compute_sam(pr(:,:,j,i),tr(:,:,j,i)));
    end
end

rmse = sqrt(sum(aux, 1)/bands);
% ERGAS
ergas = 100*ratio_ergas*sqrt(sum((rmse_per_band ./ mean_y).^2)/bands);
ergas_avg = mean(ergas);

s_avg = mean(ss,2);
avg_all = mean(s_avg);

psnr_avg = mean(ps,2);
psnr_avg_all = mean(psnr_avg);
