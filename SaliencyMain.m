clear all;
close all,
clc;

% Specify the folder containing the videos
folder = 'D:\saliency based video summarzation\Github code'; % Replace 'path_to_folder' with the actual folder path

% Get a list of all video files in the folder
videoFiles = dir(fullfile(folder, '*.mov')); % Change '*.mp4' to the appropriate video file extension

% Loop through each video file
for i = 1:length(videoFiles)
    % Read the video file
    videoPath = fullfile(folder, videoFiles(i).name);
    video = VideoReader(videoPath);
    
    % Read the first frame
    firstFrame = readFrame(video);
    
    % Initialize variables for averaging
    sumFrames = double(firstFrame);
    frameCount = 1;
    
    % Read and accumulate frames from the video
    while hasFrame(video)
        frame = readFrame(video);
        sumFrames = sumFrames + double(frame);
        frameCount = frameCount + 1;
    end
    
    % Calculate the average frame
    averageFrame = uint8(sumFrames / frameCount);

% Normalize the images
firstFrame = im2double(firstFrame);
averageFrame = im2double(averageFrame);


% for first image
[noisyR,noisyG,noisyB] = imsplit(firstFrame);
% Define the Laplacian filter.
Laplacian=[0 1 0; 1 -4 1; 0 1 0];

denoisedR = imfilter(double(noisyR),Laplacian,'same');
denoisedG = imfilter(double(noisyG),Laplacian,'same');
denoisedB = imfilter(double(noisyB),Laplacian,'same');
Filtering1 = cat(3,denoisedR,denoisedG,denoisedB);

% Apply winer filter filter
[noisyR,noisyG,noisyB] = imsplit(firstFrame);
       denoisedR = imnoise(noisyR,'gaussian',0,0.005);
     denoisedR = wiener2(denoisedR,[15 15]);
      denoisedG  = imnoise(noisyG,'gaussian',0,0.005);
    denoisedG  = wiener2(denoisedG ,[15 15]);
   denoisedB  = imnoise(noisyB,'gaussian',0,0.005);
   denoisedB  = wiener2(denoisedB ,[15 15]);
Filtering2 = cat(3,denoisedR,denoisedG,denoisedB);

% Saliency estimation
Saliency1=(Filtering1-double(Filtering2)).^2;
d1=double(firstFrame)-Filtering1;

% for 2nd image
[noisyR,noisyG,noisyB] = imsplit(averageFrame);
Laplacian=[0 1 0; 1 -4 1; 0 1 0];
denoisedR = imfilter(double(noisyR),Laplacian,'same');
denoisedG = imfilter(double(noisyG),Laplacian,'same');
denoisedB = imfilter(double(noisyB),Laplacian,'same');
b2 = cat(3,denoisedR,denoisedG,denoisedB);

% Apply winer filter filter
[noisyR,noisyG,noisyB] = imsplit(averageFrame);
       denoisedR = imnoise(noisyR,'gaussian',0,0.005);
     denoisedR = wiener2(denoisedR,[15 15]);
      denoisedG  = imnoise(noisyG,'gaussian',0,0.005);
    denoisedG  = wiener2(denoisedG ,[15 15]);
   denoisedB  = imnoise(noisyB,'gaussian',0,0.005);
   denoisedB  = wiener2(denoisedB ,[15 15]);
Image2 = cat(3,denoisedR,denoisedG,denoisedB);

% Saliency estimation
Saliency2=(b2-double(Image2)).^2;
d2=double(averageFrame)-b2;

% Weights Normalization
weight1=Saliency1./(Saliency1+Saliency2);

% Weights Normalization
weight2=Saliency2./(Saliency1+Saliency2);

Finalimag1=double(weight1).*double(d1)+double(weight2).*double(d2);

finalimage2=0.5*Filtering1+0.5*b2;

Fusedframe=double(Finalimag1)+finalimage2;
  
    Filenames = fullfile(folder, [videoFiles(i).name(1:end-4), '.jpg']);
   % imwrite(firstFrame, firstFrameFileName);
    imwrite(Fusedframe, Filenames);
    
    fprintf('First frame and average frame processsed for saliency-based fusion: %s\n', videoFiles(i).name);
end

fprintf('All videos processed.\n');
