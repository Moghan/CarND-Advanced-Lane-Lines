## Writeup for 4 - Advanced Lane Finding

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[undist_chessboard]: examples/undist_chessboard.jpg "Undistorted chessboard"
[chessboard]: camera_cal/calibration2.jpg "chessboard"
[test_frame]: examples/test_frame.jpg "unprocessed test frame"
[image_pipeline_example]: examples/image_pipeline_example.jpg "image pipeline example"
[transform_example]: examples/transform_example.jpg "image transform example"
[line_example]: examples/line_example.jpg "line example"
[final_image]: examples/final_image.jpg "final result"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###  Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

Example of chessboard calibration image:

![chessboard][chessboard]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![undistorted chessboard][undist_chessboard]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![test frame][test_frame]
#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of abs_sobel_thresh() using two different color spaces. The s-channel enhance yellow very well, but sometimes miss the white lines. In a gray image, the white lines stand out, but sometimes the yellow lines disappears. When using both, there are still areas where the yellow line is very vague and not detected by sobel. A third level of binary is added for these vague yellow lines using threshholds on the s-channel. (Code lines 310 - 326)

![pipeline example][image_pipeline_example]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

My perspective transorm is found in the `warp_image()` function (code lines 81 - 93). 

The `warp_image()` function takes as inputs an image (`img`), and an optional input for reversed transform. By measuring manually on an image with straight road, I hardcoded the source and destination points in the following manner:

```
src = np.float32([[594,450],[690,450],[1080,700],[250,700]])
dst = np.float32([[550, 0],[780, 0],[780,700],[550,700]])

```

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 594, 450      | 550, 0        | 
| 690, 450      | 780, 0      |
| 1080, 700     | 780, 700      |
| 250, 700      | 550, 700        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![warped image exapmle][transform_example]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
To find the base of the lines at the bottom of the image, I analyse the lower half in a histogram. Peaks in the histogram indicates some sort of line. To avoid mistaking the edge of the road for a line, the histogram top peak between x = 500 and its center is set to left line base. Similar is done for right line.

Then I search for pixels within a square around the line base. If enough pixels is found for a strong indication of a line, a new position is set for next square. If few pixels is found, then position for next square is set by using data earlier frames. (Code lines 114 - 179)

Using all line pixels, I fit my lane lines with a 2nd order polynomial. (Code lines 194 - 195)

![line example][line_example]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in code lines 96 - 108.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 264 through 306 in the function `fillLineMask()`.  Here is an example of my result on a test image:

![final result][final_image]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/hVcpBY3drxc)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline sobel and s-channel threshholds are very simple, and should be easy to improved. The binary made from s-channel with threshholds will make the pipeline break if the min threshhold is set too low. I suppose that with the "right" image, it will break in a similar manner.

There is no sanity check on the findings! At the moment, there will be a found line in every frame. No matter how bad the result. This works on the project_video, but it is the first I would implement in upcoming improvements.

The width of the lane is hardcoded based on knowledge of what road the vehicle is on. It would be nice if the solution could find out what kind of road the vehicle is on by itself.

The biggest problem have been time. As usual I struggle with a lot of problems, but I keep solve and learn from them, and the overall speed is defenitly increasing. The main goal at the moment is to get an approved submission.

