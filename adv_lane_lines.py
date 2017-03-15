import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob

import line

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

right_line = line.Line()
left_line = line.Line()




def _find_lines(mask):
    mask = warp_image(mask)
    
    
    global global_image_count
    global_image_count = global_image_count + 1
    # show_image(mask, mask)
    histogram = np.sum(mask[mask.shape[0]/2:, :], axis=0)

    # plt.plot(histogram)
    # plt.show()

    out_img = np.dstack((mask, mask, mask))*255

    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[500:midpoint]) + 500
    rightx_base = np.argmax(histogram[midpoint:900]) + midpoint

    # print('left:%i  right:%i' % (leftx_base, rightx_base))
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

    # print('image: %i' % global_image_count)
    # Step through the windows one by one
    for window in range(nwindows):
        margin = base_margin
        # print('win: %i , leftpos: %i , rightpos: %i , dist: %i' % (window, leftx_current, rightx_current, rightx_current - leftx_current))
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
            # print('good_left_inds (%i) > (%i) minpix' % (len(good_left_inds), minpix))
            if len(good_right_inds) > minpix:
                # print('steal from right')
                leftx_current = np.int(np.mean(nonzerox[good_right_inds]) - 230)

        if len(good_right_inds) > minpix:            
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        else:
            # print('good_right_inds (%i) > (%i) minpix' % (len(good_right_inds), minpix))
            if len(good_left_inds) > minpix:
                # print('steal from left')
                rightx_current = np.int(np.mean(nonzerox[good_left_inds]) + 230)

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
    


    # WHAT HAPPEND WITH POP !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ******** ==============

    # if(len(left_line.recent_xfitted) > 999):
    #     pop(left_line.recent_xfitted)
    #     pop(right_line.recent_xfitted)

# ----------------- TESTING -----------------------
    # all_recent_left_fits = np.hstack(left_line.recent_xfitted)
    # all_recent_right_fits = np.hstack(right_line.recent_xfitted)
    # # Extract left and right line pixel positions
    # lx = nonzerox[all_recent_left_fits]
    # ly = nonzeroy[all_recent_left_fits] 
    # rx = nonzerox[all_recent_right_fits]
    # ry = nonzeroy[all_recent_right_fits] 

    #  # Fit a second order polynomial to each
    # left_line.best_fit = np.polyfit(ly, lx, 2)
    # right_line.best_fit = np.polyfit(ry, rx, 2)

# --------------- END -- TESTING -----------------------

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    window_img = np.zeros_like(out_img)

    # plt.figure(figsize=(10,10))
    # plt.imshow(out_img)
    # for lx in right_line.recent_xfitted:

    #     plt.plot(lx, ploty, color='yellow')
    #     # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.show()


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

    left_line.radius_of_curvature = get_curverad_meters(ploty, left_line.recent_xfitted[0])
    right_line.radius_of_curvature = get_curverad_meters(ploty, right_line.recent_xfitted[0])

# -----------------------------
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(mask).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # print(left_fitx[0], left_fitx[-1])

    

    image_center = mask.shape[1] / 2

    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # right_line.line_base_pos = right_fitx[-1]
    # left_line.line_base_pos = left_fitx[-1]

    # curve_rad = (left_curverad + right_curverad) / 2
    
    

   
    # return result
    color_warp = warp_image(color_warp, unwarp = True) 

    return color_warp, result

def show_image(new , old):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(old)

    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(new , cmap='gray')
    ax2.set_title('Old Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

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
    # Calculate directional gradient
    # Apply threshold
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # s_channel = hls[:,:,2]
    s_channel = image

    if orient == 'x':
        sobel = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    if orient == 'y':
        sobel = cv2.Sobel(s_channel, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    
    abs_sobel = np.absolute(sobel)

    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    return grad_binary

# def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
#     # Calculate gradient magnitude
#     # Apply threshold
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
#     sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
#     gradmag = np.sqrt(sobelx**2 + sobely**2)
#     scale_factor = np.max(gradmag)/255 
#     gradmag = (gradmag/scale_factor).astype(np.uint8) 
#     mag_binary = np.zeros_like(gradmag)
#     mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
#     return mag_binary

# def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
#     sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
#     abs_sobelx = np.absolute(sobelx)
#     abs_sobely = np.absolute(sobely)
#     direction = np.arctan2(abs_sobely, abs_sobelx)
#     dir_binary = np.zeros_like(direction)
#     dir_binary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
#     return dir_binary

def s_channel_thresh(image, s_thresh = (170,255)):
    # s_thresh_min = 170
    # s_thresh_max = 255

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

    # print(warped.shape)
    return warped


global_image_count = 0


def get_curverad_meters(yvals, xvals):
    y_eval = np.max(yvals)
    y_eval = 719

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # # Fit new polynomials to x,y in world space
    fit_cr = np.polyfit(yvals*ym_per_pix, xvals*xm_per_pix, 2)
    # Calculate the new radii of curvature
    curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    # Now our radius of curvature is in meters
    # print(curverad, 'm')

    return curverad


def find_lines(mask):
    mask = warp_image(mask)
    
    
    global global_image_count
    global_image_count = global_image_count + 1
    # show_image(mask, mask)
    histogram = np.sum(mask[mask.shape[0]/2:, :], axis=0)

    # plt.plot(histogram)
    # plt.show()

    out_img = np.dstack((mask, mask, mask))*255

    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[500:midpoint]) + 500
    rightx_base = np.argmax(histogram[midpoint:900]) + midpoint

    # print('left:%i  right:%i' % (leftx_base, rightx_base))
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

    # print('image: %i' % global_image_count)
    # Step through the windows one by one
    for window in range(nwindows):
        margin = base_margin
        # print('win: %i , leftpos: %i , rightpos: %i , dist: %i' % (window, leftx_current, rightx_current, rightx_current - leftx_current))
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

    # plt.figure(figsize=(10,10))
    # plt.imshow(out_img)
    # for lx in right_line.recent_xfitted:

    #     plt.plot(lx, ploty, color='yellow')
    #     # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.show()


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



# test_img = mpimg.imread('camera_cal/calibration1.jpg')
# un_test = cv2.undistort(test_img, mtx, dist, None, mtx)
# show_image(un_test, test_img)


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
    # print('mask shape')
    # print(mask.shape)

    # print(left_line.current_fit)

    
    # leftx = left_line.allx
    # lefty = left_line.ally
    # rightx = right_line.allx
    # righty = right_line.ally

    leftx = np.concatenate(left_line.recent_xfitted[:numberOFLines])
    rightx = np.concatenate(right_line.recent_xfitted[:numberOFLines])

    ploty = np.linspace(0, mask.shape[0]-1, mask.shape[0] )

    lefty = np.array([])
    righty = np.array([])

    for i in range(numberOFLines):
        lefty = np.append(lefty, ploty)
        righty = np.append(righty, ploty)

    rightx = right_line.get_avarage_x(5)
    righty = ploty

    leftx = left_line.get_avarage_x(5)
    lefty = ploty

    # print(_rightx)



    out_img = np.dstack((mask, mask, mask))*255

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, mask.shape[0]-1, mask.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    right_line.recent_xfitted.insert(0, right_fitx)
    left_line.recent_xfitted.insert(0, left_fitx)
   
    # !!!!!!????? -------------------------- POP ()

    if(len(left_line.recent_xfitted) > 20):
        np.delete(left_line.recent_xfitted, -1)
        np.delete(right_line.recent_xfitted, -1)


    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # window_img = np.zeros_like(out_img)


    # # Generate a polygon to illustrate the search window area
    # # And recast the x and y points into usable format for cv2.fillPoly()
    # left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    # left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    # left_line_pts = np.hstack((left_line_window1, left_line_window2))
    # right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    # right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    # right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # # Draw the lane onto the warped blank image
    # cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    # cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    # result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)


# -----------------------------
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
    global frame
    frame += 1

    image_center = image.shape[1] / 2

    # print('original shape = ')
    # print(image.shape)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("video_frames/frame%d.jpg" % frame, image)     # save frame as JPEG file


    ksize = 7
    # print('before undistort')
    # print(image)
    image = cv2.undistort(image, mtx_Global, dist_Global, None, mtx_Global)
    # print('AFTER undistort')
    # print(image)
    # Apply each of the thresholding functions

    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    gradx_binary_s_channel = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    gradx_binary_gray = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    gradx_binary = np.zeros_like(gradx_binary_gray)
    gradx_binary[((gradx_binary_s_channel == 1) | (gradx_binary_gray == 1))] = 1



    s_binary = s_channel_thresh(image, s_thresh = (170,255))


    # grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    # mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    # dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))


    mask = np.zeros_like(s_binary)
    mask[((gradx_binary == 1) | (s_binary == 1))] = 1

    empty_mask = np.zeros_like(s_binary)

# ---------------------------------------------------------------------------

       
    # mask_fin , boxes = find_lines(mask)
    line_find_diagnostic = find_lines(mask)


    line_mask = fillLineMask(empty_mask)

    mask_fin = line_mask

    # w_mask = warp_image(mask)
    # l_mask , boxes = find_lines(w_mask)
    # mask_fin = warp_image(l_mask, unwarp = True)

# ---------------------------------------------------------------------------



    # print(image.shape)
    # print(mask_fin.shape)

    result = cv2.addWeighted(image, 1, mask_fin, 0.3, 0)

    e_mask = np.zeros_like(mask)
    color_mask = np.dstack((mask, e_mask, e_mask))

    # extract diagnose data
    curve_rad = (left_line.radius_of_curvature + right_line.radius_of_curvature) / 2

    
    # center_dist: positive = right of lane center , negative = left of lane center
    # print(image.shape[0])
    left_line.line_base_pos, right_line.line_base_pos = get_leftright_line_base(mask_fin[image.shape[0]-1,:,1])
    
    center_dist = getCenterDistMeters(image_center)
    

    # print(center_dist)

    # TESTING MOSAIC PANEL
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(result, 'Estimated lane curvature:', (30, 60), font, 0.7, (255,0,0), 2)
    cv2.putText(result, 'Estimated Meters from center:', (30, 90), font, 0.7, (255,0,0), 2)
    cv2.putText(result, '%.1f m' % curve_rad, (500, 60), font, 1, (0,255,0), 2)
    cv2.putText(result, '%.2f m' % center_dist ,(500, 90), font, 1, (0,255,0), 2)
    mainDiagScreen = line_find_diagnostic
    diag1 = color_mask
    diag9 = result


    # middle panel text example
    # using cv2 for drawing text in diagnostic pipeline.
    
    middlepanel = np.zeros((120, 1280, 3), dtype=np.uint8)
    cv2.putText(middlepanel, 'Estimated lane curvature: %s' % str(curve_rad), (30, 60), font, 0.5, (255,0,0), 2)
    cv2.putText(middlepanel, 'Estimated Meters from center: %s' % str(center_dist) ,(30, 90), font, 1, (255,0,0), 2)


    # assemble the screen example
    diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    diagScreen[0:720, 0:1280] = mainDiagScreen
    diagScreen[0:240, 1280:1600] = cv2.resize(diag1, (320,240), interpolation=cv2.INTER_AREA) 
    # diagScreen[0:240, 1600:1920] = cv2.resize(diag2, (320,240), interpolation=cv2.INTER_AREA)
    # diagScreen[240:480, 1280:1600] = cv2.resize(diag3, (320,240), interpolation=cv2.INTER_AREA)
    # diagScreen[240:480, 1600:1920] = cv2.resize(diag4, (320,240), interpolation=cv2.INTER_AREA)*4
    
    # diagScreen[600:1080, 1280:1920] = cv2.resize(diag7, (640,480), interpolation=cv2.INTER_AREA)*4
    # diagScreen[600:1080, 1280:1920] = cv2.resize(diag7, (640,480), interpolation=cv2.INTER_AREA)
    
    # diagScreen[720:840, 0:1280] = middlepanel
    # diagScreen[840:1080, 0:320] = cv2.resize(diag5, (320,240), interpolation=cv2.INTER_AREA)
    # diagScreen[840:1080, 320:640] = cv2.resize(diag6, (320,240), interpolation=cv2.INTER_AREA)
    # diagScreen[840:1080, 640:960] = cv2.resize(diag9, (320,240), interpolation=cv2.INTER_AREA)
    # diagScreen[840:1080, 960:1280] = cv2.resize(diag8, (320,240), interpolation=cv2.INTER_AREA)

    # diagScreen[720:1080, 0:640] = cv2.resize(diag5, (640,360), interpolation=cv2.INTER_AREA)
    diagScreen[720:1080, 640:1280] = cv2.resize(diag9, (640,360), interpolation=cv2.INTER_AREA)


    cv2.imwrite("diagnostic/frame%d.jpg" % frame, diagScreen)  



    # DEBUG
    # return result , mask_fin
    # return result, diagScreen

    return result





mtx_Global, dist_Global = get_camera_undistort_vals()

test_images = glob.glob('video_test_frames/*.jpg')
# test_images = glob.glob('test_images/*.jpg')


# for i in test_images:
# for i in range(26):
#     image = mpimg.imread(test_images[i])
#     image, boxes = image_pipeline(image)
#     show_image(image, boxes)

challenge_output = '3_out_vid.mp4'
clip2 = VideoFileClip('project_video.mp4')
challenge_clip = clip2.fl_image(image_pipeline)
challenge_clip.write_videofile(challenge_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))

# plt.imshow(img)


# Read in an image
# image = mpimg.imread('signs_vehicles_xygrad.png')
# image = mpimg.imread('color-shadow-example.jpg')

 









# ---------------------------------------------------------------------------------------------
# # Choose a Sobel kernel size
# ksize = 3 # Choose a larger odd number to smooth gradient measurements

# # Apply each of the thresholding functions
# gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
# # grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
# mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
# dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))


# combined = np.zeros_like(dir_binary)
# combined[((gradx == 1) & (mag_binary == 1) & (dir_binary == 1))] = 1
# # combined[(gradx == 1)] = 1

# combined = warp_image(image)
# # image = warp_image(image)
# # Run the function
# # dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))
# # Plot the result
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(image)

# ax1.set_title('Original Image', fontsize=50)
# ax2.imshow(combined, cmap='gray')
# ax2.set_title('Thresholds Combined', fontsize=50)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

# ------------------------------------------------------------------------------------



plt.show(block=True)