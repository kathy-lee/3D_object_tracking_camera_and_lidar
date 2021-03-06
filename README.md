# SFND 3D Object Tracking

This is the final project of the camera course. 


In this final project, four major tasks are completed: 
1. develop a way to match 3D objects over time by using keypoint correspondences. 
2. compute the TTC based on Lidar measurements. 
3. proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. conduct various tests with the framework. The goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor.

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.

## FP.1 Match 3D Objects

The idea of matching 3D objects is to takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property). Matches must be the ones with the highest number of keypoint correspondences.

Implementation step:
1. tranverse through each bounding box in the previous frame;
2. for each bounding box tranverse through all matched keypoint-pairs;
3. if one keypoint of this matched keypoint-pair is within this bounding box in the previous frame, then tranverse through each bounding box of the current frame to see another keypoint of this matched keypoint-pair is within it or not. If yes, then add one point to the matching score of these two bounding boxes;
4. when step 2 and step 3 finished, get the ID index with highest score which means best matched with each other;
5. add the ID pair of these two matched bounding boxes into result array.

## FP.2 Compute Lidar-based TTC

Implementation step:
1. tranverse through each point in the lidar point cloud of the previous frame, sum all the x-cordinate values together, then get its mean value;
2. tranverse through each point in the lidar point cloud of the current frame, sum all the x-cordinate values together, then get its mean value;
3. get TTC estimation through equation.


## FP.3 Associate Keypoint Correspondences with Bounding Boxes

The idea is to associate a given bounding box with the keypoints it contains.

Implementation step:
1. tranverse through the matched keypoint-pairs,if a keypoint is within the bounding box, get the norm distance between the two points in the keypoint-pair, add the distance value into an numeric array, add this keypoint-pair into an DMatch array;
2. to filter out the outliers in keypoint correspondences, get the mean value of this distance array, if the distance between a keypoint-pair is more than the mean value, filter this keypoint-pair out;
3. put the keypoint-pair array into the kptMatches member of the bounding box.


## FP.4 Compute Camera-based TTC

The implementation step:
1. tranverse through the keypoint-pair matches of previous frame and current frame, get current keypoint-pair;
2. tranverse through the keypoint-pair matches of previous frame and current frame again, get next keypoint-pair;
3. according to these two keypoint-pairs, compute distances and their distance ratio;
4. after step 1,2,3 finished, compute TTC by using median of distance ratios.

## FP.5 Performance Evaluation of Lidar_based TTC

Lidar points of the preceding vehicle through continous frames:

<img src="res/myresult.gif" width="500" height="500" />

From the lidar points we can see some outliers which may distrub the esimation of TTC. We use mean value along the x-axis to get a stable estimation.

## FP.6 Persormance Evaluation of different detector & descriptor combinations and the differences in TTC estimation

In [this spreadsheet](https://docs.google.com/spreadsheets/d/19DEHwdciBtQtau1gjFtIGvdGDedBvbBlVHXjbYBkEnE/edit?usp=sharing) we use different detector / descriptor combinations and their TTC estimation results based on lidar point cloud and camera images are shown, including the mean of TTC estimation sequence, the standard deviation of TTC estimation sequence and the min value of TTC estimation sequence during 18 frames(1.80s). According to the results, the following detector and descriptor combinations have relatively robust estimation results:
- FAST detector with BRIEF descriptor 
- FAST detector with FREAK descriptor
- FAST detector with ORB descriptor

<img src="res/myresult_2.gif" />

