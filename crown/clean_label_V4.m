%% % % % % % % % % % % % % % % % % % % %
% Xiaohui Zhang
% Nov 1st 2019
% clear labels in the projection  
% direction to delineate the 3D crown
% % % % % % % % % % % % % % % % % % % % %

% STEP ONE
% Elminate the overlap between crown to seperate the label
clear all;
clc;

fid = '/shared/anastasio1/SOMS/crown/final_processed/OA-RGND-8bit-processed/02-00-0.85';
files = dir(fid);
fids = '/shared/anastasio1/SOMS/crown/final_processed/OA-RGND-8bit-processed/02-00-0.85-correct';

for i = 3:size(files,1)
    disp(i);
    fname = fullfile(files(i).folder, files(i).name);  
    img = imread(fname);
    range = unique(img);
    if range == 0
       imwrite(img, sprintf(fullfile(fids, files(i).name)));
    else 
       img(img>range(2)) = 0;
       img(img>0)=255;
       imwrite(im2uint8(img), sprintf(fullfile(fids, files(i).name))); 
    end
    
end

%%
% STEP 2
clc;
clear;

fid = '/shared/anastasio1/SOMS/crown/final_processed/OA-RGND-8bit-processed/02-00-0.85-correct';
files = dir(fid);
fids = '/shared/anastasio1/SOMS/crown/final_processed/OA-RGND-8bit-processed/02-00-0.85-correct-test';

% initialize the first slice because we are correcting the labels in the
% order of the labels in the bottom slice
top_img = imbinarize(imread(fullfile(files(3).folder, files(3).name)));

% define element for image openning operation
SE = strel('diamond',10);
top_img = imopen(top_img,SE);

for i = 3:(size(files,1)-1)
    disp(i);
    
    % read, binarize and run opening operation on image
    bottom_fname = fullfile(files(i+1).folder, files(i+1).name);  
    bottom_img = imbinarize(imread(bottom_fname));
    bottom_img = imopen(bottom_img,SE);
    

    % do the same for the top slice
    L_t = bwlabel(top_img);
    top_stats = regionprops(top_img,'PixelList','PixelIdxList');
    top_vec = reshape(L_t, [numel(L_t), 1]);
    
    % label the resulted bottom slice and find the pixel index for each label 
    % in bottom image
    L_b = bwlabel(bottom_img);
    bottom_stats = regionprops(bottom_img,'PixelList','PixelIdxList');  
    bottom_vec = reshape(L_b, [numel(L_b), 1]);
    
    if ~isempty(top_stats) && ~isempty(bottom_stats)    

        for label_index = 1:size(bottom_stats,1)
        
            % calculate how many crown labels are overlaid by each label in the bottom slice
            num_overlap = unique(top_vec(bottom_stats(label_index).PixelIdxList));
            num_overlap = num_overlap(num_overlap~=0);
        
            % if there's overlap in the projection between top and bottom slice
            % compare the area of multiple overlap and decide which label area
            % need to be erased
            if numel(num_overlap)>1
                num_label = 0;
                for j = 1:numel(num_overlap)
                    num_label_temp = sum(top_vec(bottom_stats(label_index).PixelIdxList) == num_overlap(j));
                    if num_label_temp>num_label
                        kept_label = num_overlap(j);
                        erase_label = setdiff(num_overlap,kept_label);
                    end
                    num_label = num_label_temp;
                end
            
                % break the overlap across multiple crowns (STEP 1)
                bovert = top_vec(bottom_stats(label_index).PixelIdxList);
                a = bottom_stats(label_index).PixelIdxList;
                z = find(bovert==0);
                z_part = a(z);
                bottom_vec(z_part)=0;
                top_vec(z_part)=0;
            
                % also move the overlap between labels and background to make
                % sure that the projection will not be the adjacent two crowns
                for k = 1:numel(erase_label)
                    nz = find(bovert==erase_label(k));
                    nz_part = a(nz);
                    bottom_vec(nz_part)=0;
                    top_vec(nz_part)=0;
                end

            end
        
            new_bottom_img = reshape(bottom_vec,[size(bottom_img,1), size(bottom_img,2)]);
            new_top_img = reshape(top_vec,[size(top_img,1), size(top_img,2)]);
        
            bottom_vec = reshape(new_bottom_img, [numel(new_bottom_img), 1]);
            top_vec = reshape(new_top_img, [numel(new_top_img), 1]);
        
            % for these special crowns being processed, only keep the shared
            % area between top and bottom slice to delineate them in projected
            % x-y plane using "and" operation (STEP 2)
            bpt_img = imbinarize(new_bottom_img) & imbinarize(new_top_img);
            bpt_vec = reshape(bpt_img,[numel(bpt_img),1]);
        
            if any(bpt_vec(bottom_stats(label_index).PixelIdxList))
                bottom_vec(bottom_stats(label_index).PixelIdxList) = 0;
                new_labelb_index = find(bpt_vec(bottom_stats(label_index).PixelIdxList)==1);
                b = bottom_stats(label_index).PixelIdxList;
                bottom_vec(b(new_labelb_index)) = label_index;
            end
        
            % fine the corresponding label index in top slice for each label
            % index in bottom slice
            top_indexs = unique(top_vec(bottom_stats(label_index).PixelIdxList));
            top_indexs = top_indexs(top_indexs~=0);   
        
            % if for the label in bottom slice there's no match in top
            % slice, skip this step
            if numel(top_indexs)~=0
                top_gap_index = find(top_vec==top_indexs);
                top_vec(top_gap_index) = 0;
                new_labelt_index = find(bpt_vec(top_gap_index)==1);
                top_vec(top_gap_index(new_labelt_index)) = top_indexs;
            end
        
            new_bottom_img = reshape(bottom_vec,[size(bottom_img,1), size(bottom_img,2)]);
            new_top_img = reshape(top_vec,[size(top_img,1), size(top_img,2)]);
        
        % update top and bottom slice
            bottom_img = new_bottom_img;
            top_img = new_top_img;
        end
    
        % update the top slice with the new bottom slice
        top_img = new_bottom_img;
   
        % save the updated images
        imwrite(im2uint8(new_bottom_img), sprintf(fullfile(fids, files(i+1).name))); 
        imwrite(im2uint8(new_top_img), sprintf(fullfile(fids, files(i).name))); 
    else
        new_bottom_img = bottom_img;
        new_top_img = top_img;
        
        top_img = new_bottom_img;      
        imwrite(im2uint8(new_bottom_img), sprintf(fullfile(fids, files(i+1).name))); 
        imwrite(im2uint8(new_top_img), sprintf(fullfile(fids, files(i).name))); 
    end
end

