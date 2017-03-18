import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

import line

from moviepy.editor import VideoFileClip


IS_DEBUG = False

right_line = line.Line()
left_line = line.Line()

# Used for DEBUG
def show_image(new , old):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 5))
    f.tight_layout()
    ax1.imshow(old)

    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(new)
    ax2.set_title('Final result', fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.05)


def get_camera_undistort_vals():
    CALIBRATION_IMAGE_PATH = 'camera_cal/'
    cal_images = glob.glob(CALIBRATION_IMAGE_PATH + 'calibration*.jpg')

    objpoints = []
    imgpoints = []

    objp = np.zeros((6*9, 3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    for ci in cal_images:
        print('working on ' + ci)
        img = mpimg.imread(ci)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist 

def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    if orient == 'x':
        sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    if orient == 'y':
        sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    
    abs_sobel = np.absolute(sobel)

    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    return grad_binary


def s_channel_thresh(image, s_thresh = (85,255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    return s_binary


def warp_image(img, unwarp = False):
    image_size = (img.shape[1], img.shape[0])
    src = np.float32([[594,450],[690,450],[1080,700],[250,700]])
    dst = np.float32([[550, 0],[780, 0],[780,700],[550,700]])

    if unwarp:
        Minv = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(img, Minv, image_size, flags=cv2.INTER_LINEAR)
    else:
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, image_size, flags=cv2.INTER_LINEAR)

    return warped


def get_curverad_meters(yvals, xvals):
    y_eval = np.max(yvals)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # # Fit new polynomials to x,y in world space
    fit_cr = np.polyfit(yvals*ym_per_pix, xvals*xm_per_pix, 2)
    # Calculate the new radii of curvature
    curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])

    return curverad


def find_lines(mask):
    mask = warp_image(mask)

    histogram = np.sum(mask[mask.shape[0]/2:, :], axis=0)

    # create image for diagnostics and debug features
    out_img = np.dstack((mask, mask, mask))*255

    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[500:midpoint]) + 500
    rightx_base = np.argmax(histogram[midpoint:900]) + midpoint

    # Choose the number of sliding windows
    nwindows = 14
    # Set height of windows
    window_height = np.int(mask.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = mask.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    base_margin = 40
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        margin = base_margin
        # Identify window boundaries in x and y (and right and left)
        win_y_low = mask.shape[0] - (window+1)*window_height
        win_y_high = mask.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        else:
            # If to few pixels is found, use previous line to find a useful location
            if(len(left_line.recent_xfitted) > 0):
                leftx_current = int(left_line.recent_xfitted[0][win_y_high-1])

        if len(good_right_inds) > minpix:            
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        else:
            if (len(right_line.recent_xfitted) > 0):
                rightx_current = int(right_line.recent_xfitted[0][win_y_high-1])


    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    left_line.allx = leftx
    left_line.ally = lefty
    right_line.allx = rightx
    right_line.ally = righty

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    left_line.current_fit = left_fit
    right_line.current_fit = right_fit

    # Generate x and y values for plotting
    ploty = np.linspace(0, mask.shape[0]-1, mask.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    right_line.recent_xfitted.insert(0, right_fitx)
    left_line.recent_xfitted.insert(0, left_fitx)

    left_line.radius_of_curvature = get_curverad_meters(ploty, left_line.recent_xfitted[0])
    right_line.radius_of_curvature = get_curverad_meters(ploty, right_line.recent_xfitted[0])
   

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    window_img = np.zeros_like(out_img)
    

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    return result


def get_leftright_line_base(pts):
    nonzero = pts.nonzero()

    leftx = nonzero[0][0]
    rightx = nonzero[0][-1]

    return leftx, rightx


def getCenterDistMeters(image_center):
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    lane_center = left_line.line_base_pos + (right_line.line_base_pos - left_line.line_base_pos) / 2
    center_dist = (image_center - lane_center) * xm_per_pix

    return center_dist


def fillLineMask(mask, numberOFLines = 1):
    ploty = np.linspace(0, mask.shape[0]-1, mask.shape[0] )

    rightx = right_line.get_avarage_x(5)
    righty = ploty

    leftx = left_line.get_avarage_x(5)
    lefty = ploty

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, mask.shape[0]-1, mask.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    right_line.recent_xfitted.insert(0, right_fitx)
    left_line.recent_xfitted.insert(0, left_fitx)
   
    if(len(left_line.recent_xfitted) > 20):
        np.delete(left_line.recent_xfitted, -1)
        np.delete(right_line.recent_xfitted, -1)
   
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(mask).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
   
    # return result
    color_warp = warp_image(color_warp, unwarp = True) 
    
    return color_warp


frame = 0
def image_pipeline(image):
    # counting frames for DEBUG features
    global frame
    frame += 1

    image_center = image.shape[1] / 2

    # kernel for abs_sobel_thresh()
    ksize = 3
    
    image = cv2.undistort(image, mtx_Global, dist_Global, None, mtx_Global)


    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    # Apply abs_sobel_thresh on s_channel (for yellow lines) and gray (for white lines)
    gradx_binary_s_channel = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=(20, 200))
    gradx_binary_gray = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(20, 200))

    gradx_binary = np.zeros_like(gradx_binary_gray)
    

    gradx_binary[((gradx_binary_s_channel == 1) | (gradx_binary_gray == 1))] = 1
    s_binary = s_channel_thresh(image, s_thresh = (95,255))


    mask = np.zeros_like(s_binary)
    mask[((gradx_binary == 1) | (s_binary == 1))] = 1

    # Collected data of found lines is stored in the Line objects left_line and right_line
    line_find_diagnostic = find_lines(mask)

    empty_mask = np.zeros_like(s_binary)
    line_mask = fillLineMask(empty_mask)

    result = cv2.addWeighted(image, 1, line_mask, 0.3, 0)

    # extract diagnose data
    curve_rad = (left_line.radius_of_curvature + right_line.radius_of_curvature) / 2
    left_line.line_base_pos, right_line.line_base_pos = get_leftright_line_base(line_mask[image.shape[0]-1,:,1])
    center_dist = getCenterDistMeters(image_center)


    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(result, 'Estimated lane curvature:', (30, 60), font, 0.7, (255,0,0), 2)
    cv2.putText(result, 'Estimated Meters from center:', (30, 90), font, 0.7, (255,0,0), 2)
    cv2.putText(result, '%.1f m' % curve_rad, (500, 60), font, 1, (0,255,0), 2)
    cv2.putText(result, '%.2f m' % center_dist ,(500, 90), font, 1, (0,255,0), 2)
    
    
# MOSAIC PANEL --------------- FOR DEBUG and diagnostic
# Found this diagnostic view on the forum. Great help!

    diag5 = image
    diag9 = result
    mainDiagScreen = line_find_diagnostic
    
    # assemble the screen example
    diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    diagScreen[0:720, 0:1280] = mainDiagScreen
    # diagScreen[0:240, 1280:1600] = cv2.resize(diag1, (320,240), interpolation=cv2.INTER_AREA) 
    # diagScreen[0:240, 1600:1920] = cv2.resize(diag2, (320,240), interpolation=cv2.INTER_AREA)
    # diagScreen[240:480, 1280:1600] = cv2.resize(diag3, (320,240), interpolation=cv2.INTER_AREA)
    # diagScreen[240:480, 1600:1920] = cv2.resize(diag4, (320,240), interpolation=cv2.INTER_AREA)*4
    
    # diagScreen[600:1080, 1280:1920] = cv2.resize(diag7, (640,480), interpolation=cv2.INTER_AREA)*4
    # diagScreen[600:1080, 1280:1920] = cv2.resize(diag7, (640,480), interpolation=cv2.INTER_AREA)
    
    diagScreen[720:1080, 0:640] = cv2.resize(diag5, (640,360), interpolation=cv2.INTER_AREA)
    diagScreen[720:1080, 640:1280] = cv2.resize(diag9, (640,360), interpolation=cv2.INTER_AREA)

# END MOSAIC PANEL

    if IS_DEBUG:
        result = diagScreen

    return result

mtx_Global, dist_Global = get_camera_undistort_vals()

test_images = glob.glob('video_test_frames/*.jpg')




challenge_output = 'advanced_lane_finding_video.mp4'
clip2 = VideoFileClip('project_video.mp4')
challenge_clip = clip2.fl_image(image_pipeline)
challenge_clip.write_videofile(challenge_output, audio=False)


plt.show(block=True)