################# P3 Advanced Lane Detection  ##################
#
#                    Done by Wael Farag                        #
################################################################

# Import the used Libraries
import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

# Some Initializations for parameters:

Test_images_path = "./test_images/"
Output_path = "./output_images/"
Camera_Cal_images_path = "./"
Data_path = "C:/work/MyFiles/Research/DeepLearning/CarND/AdvLaneDetect/CarND-Advanced-Lane-Lines-master/data/"
Video_path = "./"

# Flag to disable any verbosity or display during Video processing
is_Video = True
#is_Video = False

# Flag to indicate if the Camera Calibration Data is already available or
# needs run the calibration routines
Camera_is_Calibrated = True
#Camera_is_Calibrated = False

frame_count = 0   # Video frame counter

#================================================================================#
# Define a class to receive the characteristics of each line detection
# Define a Line() class to keep track of all the interesting parameters measured
# from frame to frame

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients for the whole fits
        self.all_fits = []
        #polynomial coefficients for the most recent fit
        self.current_fit = np.array([0,0,0], dtype='float')
        # polynomial coefficients for the previous fit
        self.previous_fit = np.array([0,0,0], dtype='float')
        #radius of curvature of the line in some units
        self.radius_of_curvature = []
        #distance in meters of vehicle center from the line
        self.line_base_pos = []
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx_current_fit = None
        #y values for detected line pixels
        self.ally_current_fit = None

#============================= End of Class Definition===========================#
#===============================================================================#
#                      Camera Calibration                                       #
#===============================================================================#

if not Camera_is_Calibrated:
    # Number of points in the Calibration image
    N_Corners_x = 9  # the number of inside corners in x direction
    N_Corners_y = 6  # the number of inside corners in y direction

    #  Array to store object points and real points from all images
    ObjectPoints = []       # 3D points in real world space
    ImagePoints  = []       # 2D points in image plan

    #  prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(9,5,0)
    ObjPoints = np.zeros((N_Corners_x*N_Corners_y,3), np.float32)
    ObjPoints[:,:2] = np.mgrid[0:N_Corners_x,0:N_Corners_y].T.reshape(-1,2)     # X & Y Coordinates

    #  Prepare the drawing stage
    plt.figure(figsize=(16, 12))
    #plt.subplots(3,6, figsize=(30, 20))
    #plt.title("    The Chess Board images")
    #print("    The Chess Board images")

    plot_index = 0

    for image_file_name in os.listdir(Camera_Cal_images_path):
        # read the Camera Calibration Images
        # print(image_file_name)
        camera_cal_color_img = mpimg.imread(Camera_Cal_images_path + image_file_name)
        camera_cal_gray_img = cv2.cvtColor(camera_cal_color_img, cv2.COLOR_RGB2GRAY)
        # plt.imshow(camera_cal_gray_img, cmap='gray')
        # plt.show()

        #  Find the chessboard corners in the Camera Calibration gray images
        #  retValue => reurn value True or False
        #  Cal_img_corners => Matrix (N_Corners_x by N_Corners_y) of detected corner points
        retValue, Cal_img_corners = cv2.findChessboardCorners(camera_cal_gray_img, (N_Corners_x, N_Corners_y), None)
        # If corners are found, draw the corners on the original image
        if retValue == True:
            ImagePoints.append(Cal_img_corners)
            ObjectPoints.append(ObjPoints)
            # Draw and display the corners
            camera_cal_grid_img = cv2.drawChessboardCorners(camera_cal_color_img,
                                                            (N_Corners_x, N_Corners_y), Cal_img_corners, retValue)
            plt.subplot(4, 5, plot_index + 1)
            plt.imshow(camera_cal_grid_img)
            # fig_axis[plot_index].axis('off')
            # fig_axis[plot_index]
            plot_index = plot_index + 1

    # Visualize all calibration images with grids
    plt.show()

    # Load and use an image that is not used in Calibration
    Test_cal_img = mpimg.imread(Camera_Cal_images_path + "calibration01.jpg")
    Test_cal_img_gray = cv2.cvtColor(Test_cal_img, cv2.COLOR_RGB2GRAY)
    Cal_image_size = np.shape(Test_cal_img_gray)

    # Do camera calibration given object points and image points
    retValue, Camera_Matrix, Distortion_Coefficients, rvecs, tvecs = \
        cv2.calibrateCamera( ObjectPoints, ImagePoints, Cal_image_size , None, None)
    undistored_image = cv2.undistort(Test_cal_img, Camera_Matrix,
                                     Distortion_Coefficients, None, Camera_Matrix)

    # Draw both the Distorted and Undistorted Image
    plt.figure(figsize=(16, 12))
    plt.subplot(1,2,1)
    plt.imshow(Test_cal_img)
    plt.title("Original Image", fontsize=15)
    plt.subplot(1,2,2)
    plt.imshow(undistored_image)
    plt.title("Undistorted Image", fontsize=15)
    plt.show()

    # Save the Camera Matrix and Distortion Coefficients for later use
    # Save in "pickle" file
    Camera_Data = {}
    Camera_Data["mtx"] = Camera_Matrix
    Camera_Data["dist"] = Distortion_Coefficients
    pickle.dump( Camera_Data, open( Data_path + "camera_calibration.p", "wb" ) )
#================================================================================
#==================== End - Camera Calibration - ================================
#================================================================================
# If the Camera is already Calibrated, Just load the camera data
#=================== Loading Camera Calibration Data ============================#
if Camera_is_Calibrated:
    dist_pickle = pickle.load( open( Data_path + "camera_calibration.p" , "rb" ) )
    Camera_Matrix = dist_pickle["mtx"]
    Distortion_Coefficients = dist_pickle["dist"]

#================================================================================#
#    Define The Warp Function
#=================================================================================
def Warp(img, src_points, dist_points):
    image_size = np.shape(img)
    # Get M, the perspective transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src_points, dist_points)
    Minv = cv2.getPerspectiveTransform(dist_points, src_points)
    # Warp the recieved image to a top-down view
    Warped_image = cv2.warpPerspective(img, M, (image_size[1], image_size[0]), flags=cv2.INTER_LINEAR)
    return Warped_image, M, Minv
#=========================== End of Defintion of the Warp Function ================
#==================================================================================
#================= Start Defining the Sobel / Gradient functions ===================
#==================================================================================
# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def Abs_Sobel_Thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    else:
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output


# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def Mag_Thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


# Define a function to threshold an image for a given range and Sobel kernel
def Dir_Threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output
#========================= End Sobel / Gradient function  ============================
#=====================================================================================
#=====================================================================================
#========== Start Defining HLS & HSV Color Maps / Color Selection functions ==========
#=====================================================================================
def HLS_select_S_Channel(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def HSV_select_V_Channel(img, thresh=(0, 255)):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    V_channel = hsv[:,:,2]
    binary_output = np.zeros_like(V_channel)
    binary_output[(V_channel > thresh[0]) & (V_channel <= thresh[1])] = 1
    return binary_output

def HSV_select_Yellow(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower = np.array([20, 60, 60])
    upper = np.array([38, 174, 250])
    mask = cv2.inRange(hsv, lower, upper)
    binary_output = np.zeros_like(mask)
    binary_output[mask > 0] = 1
    return binary_output

def RGB_select_White(img):
    lower = np.array([202, 202, 202])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(img, lower, upper)
    binary_output = np.zeros_like(mask)
    binary_output[mask > 0] = 1
    return binary_output

def LAB_select_B_Channel(img, thresh=(0, 255)):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    b_channel = lab[:,:,2]
    binary_output = np.zeros_like(b_channel)
    binary_output[(b_channel > thresh[0]) & (b_channel <= thresh[1])] = 1
    return binary_output

def HSV_select_S_Channel(img, thresh=(0, 255)):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    S_channel = hsv[:,:,1]
    binary_output = np.zeros_like(S_channel)
    binary_output[(S_channel > thresh[0]) & (S_channel <= thresh[1])] = 1
    return binary_output

def LUV_select_L_Channel(img, thresh=(0, 255)):
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    L_channel = luv[:,:,0]
    binary_output = np.zeros_like(L_channel)
    binary_output[(L_channel > thresh[0]) & (L_channel <= thresh[1])] = 1
    return binary_output

def YUV_select_Y_Channel(img, thresh=(0, 255)):
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    Y_channel = yuv[:,:,0]
    binary_output = np.zeros_like(Y_channel)
    binary_output[(Y_channel > thresh[0]) & (Y_channel <= thresh[1])] = 1
    return binary_output
#=======================  End - Defining HLS & HSV Color Map   ======================
#====================================================================================
#      Start next the image processing pipeline for detecting the Lane Line
#      Using S Channel Selection and Sobel transforms
#====================================================================================
def Color_Gradients_pipeline(img):
    # Convert imagae to HLS Color domain and select the S channel
    # hls_S_channel = HLS_select_S_Channel(img, thresh=(170, 255)).astype(np.float)
    # Convert imagae to LAB Color domain and select the B channel
    lab_B_channel = LAB_select_B_Channel(img, thresh=(170, 255)).astype(np.float)
    # Select Yellow Color from HSV color map
    Yellow_binary = HSV_select_Yellow(img)
    # Select Lumina 'Y' channel from YUV color map
    yuv_Y_channel = YUV_select_Y_Channel(img, thresh=(200, 255)).astype(np.float)
    # Select White Color from RGB color map
    White_binary = RGB_select_White(img)
    # Convert imagae to HSV Color domain and select the S channel
    # hsv_S_channel = HSV_select_S_Channel(img, thresh=(170, 255)).astype(np.float)
    # Convert imagae to LUV Color domain and select the L channel
    luv_L_channel = LUV_select_L_Channel(img, thresh=(200, 255)).astype(np.float)
    # Sobel Transforms
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx_binary    = Abs_Sobel_Thresh(img, orient='x', thresh_min=20, thresh_max=220)
    abs_sobely_binary    = Abs_Sobel_Thresh(img, orient='y', thresh_min= 150, thresh_max=180)
    mag_threshold_binary = Mag_Thresh(img, sobel_kernel=7, mag_thresh=(120, 200))
    dir_threshold_binary = Dir_Threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
    combined_sobel = np.zeros_like(abs_sobelx_binary)
    combined_sobel[((abs_sobelx_binary == 1) & (abs_sobely_binary == 1))
             | ((mag_threshold_binary == 1) & (dir_threshold_binary == 1))] = 1

    # Combine both the results of the L, B & Y Channels slection and Sobel Gradients
    combined_sobel_l_b_y_channel = np.zeros_like(abs_sobelx_binary)
    combined_sobel_l_b_y_channel[(luv_L_channel == 1) | (lab_B_channel == 1)
                                 | (combined_sobel == 1) | (yuv_Y_channel == 1)] = 1

    # Combine both the White and Yellow binary images
    combined_Yellow_White = np.zeros_like(abs_sobelx_binary)
    combined_Yellow_White[(Yellow_binary == 1) | (White_binary == 1)] = 1

    # Combine all the binary images
    combined_all = combined_sobel_l_b_y_channel | combined_Yellow_White

    #return_binary
    return combined_all, combined_Yellow_White, combined_sobel_l_b_y_channel, \
           luv_L_channel, lab_B_channel, yuv_Y_channel
#=====================================================================================
#=============  Drawing two Images Function ==========================================
#=====================================================================================
def Display_Two_Adjacent_Images(img1, title1, setting1, img2, title2, setting2):
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap = setting1)
    plt.title(title1, fontsize=15)
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap = setting2)
    plt.title(title2, fontsize=15)
    plt.show()
#=====================================================================================
# Need to decide explicitly which pixels are part of the lines and which belong to the
# left line and which belong to the right line.
# First take a histogram along all the columns in the lower half of the image
# Search for the lines using a sliding window, placed around the line centers
#=====================================================================================
def Display_Lane_ploy(hist, img, left_fit, right_fit):
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    plt.figure(figsize=(20, 10))
    plt.subplot(1,2,1)
    plt.plot(hist)
    plt.title('Histogram')
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.title('Lane Lines with Sliding Windows and Fit Polynomials')
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()


def Locate_Lane_Lines_Fit_Polynomial(Warped_masked_image):

    # Take a histogram of the bottom half of the image
    mid_array_index = np.int(Warped_masked_image.shape[0] / 2)
    histogram = np.sum(Warped_masked_image[mid_array_index:, :], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((Warped_masked_image, Warped_masked_image, Warped_masked_image)) * 255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 10
    # Set height of windows
    window_height = np.int(Warped_masked_image.shape[0] / nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = Warped_masked_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Initialize Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    if is_Video:
        # Set the width of the windows +/- margin
        margin = 90
        # Set minimum number of pixels found to recenter window
        minpix = 30
    else:
        # Set the width of the windows +/- margin
        margin = 60
        # Set minimum number of pixels found to recenter window
        minpix = 25

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = Warped_masked_image.shape[0] - (window + 1) * window_height
        win_y_high = Warped_masked_image.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                      (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each lane
    if lefty != [] and leftx != []:
        left_fit = np.polyfit(lefty, leftx, 2)
        Left_Lane.detected = True
        Left_Lane.previous_fit = Left_Lane.current_fit
        Left_Lane.current_fit = left_fit
        Left_Lane.all_fits.append(Left_Lane.current_fit)
    else:
        left_fit = Left_Lane.current_fit
        Left_Lane.detected = False

    if righty != [] and rightx != []:
        right_fit = np.polyfit(righty, rightx, 2)
        Right_Lane.detected = True
        Right_Lane.previous_fit = Right_Lane.current_fit
        Right_Lane.current_fit = right_fit
        Right_Lane.all_fits.append(Right_Lane.current_fit)
    else:
        right_fit = Right_Lane.current_fit
        Right_Lane.detected = False

    # print('\n','left_fit', left_fit)
    # print('right_fit', right_fit)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return left_fit, right_fit, leftx, lefty, rightx, righty, out_img, histogram

#=====================================================================================
#============= Search in a margin around the previous line position ==================
#=====================================================================================
def Recursive_Search_Lane_Lines(out_img, left_fit, right_fit):
    nonzero = out_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    out_img = np.dstack((out_img, out_img, out_img)) * 255
    # Create an image to draw on and an image to show the selection window
    window_img = np.zeros_like(out_img)
    ploty = np.linspace(0, out_img.shape[0] - 1, out_img.shape[0])

    margin = 90

    if left_fit != [] and right_fit != []:
        left_lane_inds = \
        ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = \
        ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[2] + margin)))
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit, right_fit = ([0, 0, 0], [0, 0, 0])
        left_fitx, right_fitx = ([], [])
        left_line_pts, right_line_pts = ([],[])

        if leftx != [] and lefty != []:
            left_fit = np.polyfit(lefty, leftx, 2)
            if is_Video:     # Smoothing by averaging over the last 3 fits
                left_fit = (left_fit + Left_Lane.current_fit + Left_Lane.previous_fit) / 3

            # Sanity check for the new calculated left_fit compared to the previous calculated one
            if is_Video:
                Left_Lane.diffs = Left_Lane.current_fit - left_fit
                if abs(Left_Lane.diffs[0]) > 0.0005 or abs(Left_Lane.diffs[1]) > 0.7 or abs(Left_Lane.diffs[1]) > 80.0:
                    left_fit = Left_Lane.current_fit
                else:
                    Left_Lane.best_fit = left_fit

            # Generate x and y values for plotting
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                            ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))

        if rightx != [] and righty != []:
            right_fit = np.polyfit(righty, rightx, 2)
            if is_Video:          # Smoothing by averaging over the last 3 fits
                right_fit = (right_fit + Right_Lane.current_fit + Right_Lane.previous_fit) / 3

            # Sanity check for the new calculated right_fit compared to the previous calculated one
            if is_Video:
                Right_Lane.diffs = Right_Lane.current_fit - right_fit
                if abs(Right_Lane.diffs[0]) > 0.0005 or abs(Right_Lane.diffs[1]) > 0.7 or abs(Right_Lane.diffs[1]) > 80.0:
                    right_fit = Right_Lane.current_fit
                else:
                    Right_Lane.best_fit = right_fit

            # Generate x and y values for plotting
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                         ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Sanity Check - reject mysterious lane fits
        # if (right_fit[2]- left_fit[2]) < 400 :
        if left_fit[2] > right_fit[2]:
            left_fit  = Left_Lane.current_fit
            right_fit = Right_Lane.current_fit

        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        # Generate a polygon to illustrate the search window area

        # Draw the lane onto the warped blank image
        if not is_Video:
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))

        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        Left_Lane.detected  = True
        Right_Lane.detected = True

        Left_Lane.previous_fit  = Left_Lane.current_fit
        Right_Lane.previous_fit = Right_Lane.current_fit
        Left_Lane.current_fit  = left_fit
        Right_Lane.current_fit = right_fit
        Left_Lane.all_fits.append(Left_Lane.current_fit)
        Right_Lane.all_fits.append(Right_Lane.current_fit)
    else:
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        left_fitx, right_fitx = ([], [])
        leftx, lefty          = ([], [])
        rightx, righty        = ([], [])

        left_fit  = Left_Lane.current_fit
        right_fit = Right_Lane.current_fit
        Left_Lane.detected  = False
        Right_Lane.detected = False

    # Record the x & y values for both left and right lanes
    Left_Lane.allx_current_fit  = left_fitx
    Right_Lane.allx_current_fit = right_fitx
    Left_Lane.ally_current_fit  = ploty
    Right_Lane.ally_current_fit = ploty

    if not is_Video:
        plt.figure(figsize=(20, 10))
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.title('Lane Lines with Fine Search and Fit Polynomials')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    return left_fit, right_fit, leftx, lefty, rightx, righty, result
#=====================================================================================
#==================== Left & Right Lanes Curvature Measurement =======================
#=====================================================================================
def Measure_Curvature_and_Center_Dist(img, left_fit, right_fit, leftx, lefty, rightx, righty):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    img_hieght = img.shape[0] - 1

    if leftx != [] and rightx != []:
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * img_hieght * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) \
                        / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * img_hieght * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) \
                         / np.absolute(2 * right_fit_cr[0])
    else:
        left_curverad  = 0
        right_curverad = 0

    Left_Lane.radius_of_curvature.append(left_curverad)
    Right_Lane.radius_of_curvature.append(right_curverad)

    if leftx != [] and rightx != []:
        # The Car position which is always in the middle of the width of the image
        # It is used as a reference to measure the deviation of the lane center from it
        Car_reference_position = np.shape(img)[1]/2
        # The values of the intercepts with the X axis for both Left and Right lanes.
        X_left_intercept = left_fit[0]*(img_hieght**2) + left_fit[1]*img_hieght + left_fit[2]
        X_right_intercept = right_fit[0] * (img_hieght**2) + right_fit[1] * img_hieght + right_fit[2]
        # Calculating the center of the lane on the X axis
        Actual_Lane_Center = (X_right_intercept + X_left_intercept)/2
        Car_Dist_from_Lane_Center = (Car_reference_position - Actual_Lane_Center) * xm_per_pix
        Left_Lane.line_base_pos.append(Car_Dist_from_Lane_Center - X_left_intercept)    # should be always a positive value
        Right_Lane.line_base_pos.append(X_right_intercept - Car_Dist_from_Lane_Center)  # should be always a positive value
    else:
        Car_Dist_from_Lane_Center = 0

    return left_curverad, right_curverad, Car_Dist_from_Lane_Center
#=================================================================================================
#===================  Draw identified lane filled with green color ===============================
#=================================================================================================
def HighLight_Lane(undist_img, Warped_binary_image, left_fit, right_fit, Minv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(Warped_binary_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    h = Warped_binary_image.shape[0]
    ploty = np.linspace(0, h - 1, h)

    if left_fit != [] and right_fit != []:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (Warped_binary_image.shape[1], Warped_binary_image.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undist_img, 1, newwarp, 0.3, 0)
    else:
        newwarp = cv2.warpPerspective(color_warp, Minv, (Warped_binary_image.shape[1], Warped_binary_image.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undist_img, 1, newwarp, 0.3, 0)

    return result

#=====================================================================================
#======================= Define Process Image Function ===============================
#=====================================================================================
def process_image_lane(Test_img):

    global frame_count

    undistorted_Test_img = cv2.undistort(Test_img, Camera_Matrix,
                                         Distortion_Coefficients, None, Camera_Matrix)
    # Draw both the Distorted and Undistorted Images
    if not is_Video:
        Display_Two_Adjacent_Images(Test_img, "Original Test Image", None
                                , undistorted_Test_img, "Undistorted Test Image", None)

    # Call the image processing pipeline
    Combined_all_result, Combined_Yellow_White_result, Color_Gradients_result, L_channel_img, B_channel_img, \
    Y_channel_img = Color_Gradients_pipeline(undistorted_Test_img)
    # Plot the result of applying Color mappings like HLS & HSV and Sobel Gradients
    # on the undistorted image
    if not is_Video:
        Display_Two_Adjacent_Images(L_channel_img, "L Channel Image", 'gray',
                                    B_channel_img, "B Channel Image", 'gray')
        Display_Two_Adjacent_Images(Y_channel_img, "Y Channel Image", 'gray',
                                    Color_Gradients_result, "Applying Color mapping & Sobel gradients pipline", 'gray')
        Display_Two_Adjacent_Images(Combined_Yellow_White_result, "White and Yellow Colored Masked", 'gray',
                                    Combined_all_result, "All Techniques Combined", 'gray')
    # defining the Region Of Intetret (ROI)
    ROI_Vertix_upper_left  = (620, 420)  # (x , y)
    ROI_Vertix_upper_right = (680, 420)
    ROI_Vertix_lower_right = (1200, 720)
    ROI_Vertix_lower_left  = (150, 720)
    Region_of_Interest_Vertices = np.array(
        [[ROI_Vertix_upper_left, ROI_Vertix_upper_right, ROI_Vertix_lower_right, ROI_Vertix_lower_left]])

    # Create a black image
    ROI_img = np.zeros_like(Color_Gradients_result)
    # print(np.shape(img1))

    # Draw a diagonal red line with thickness of 5 px
    cv2.line(ROI_img, ROI_Vertix_upper_left, ROI_Vertix_upper_right, (255, 0, 0), 5)
    cv2.line(ROI_img, ROI_Vertix_upper_right, ROI_Vertix_lower_right, (255, 0, 0), 5)
    cv2.line(ROI_img, ROI_Vertix_lower_right, ROI_Vertix_lower_left, (255, 0, 0), 5)
    cv2.line(ROI_img, ROI_Vertix_lower_left, ROI_Vertix_upper_left, (255, 0, 0), 5)

    ROI_Color_Gredient_result = cv2.addWeighted(255 * Color_Gradients_result, 0.8, ROI_img, 1, 0)

    if not is_Video:
        Display_Two_Adjacent_Images(ROI_img, "Region of Interest", None,
                                ROI_Color_Gredient_result, "Using ROI on the Color & Gradients result", None)

    # Mask the parts in the image other than the ROI
    Black_MASK = np.zeros_like(Color_Gradients_result)  # Construct a Black MASK
    Color_of_MASK_Area_to_Ignore = 255  # White Color
    cv2.fillPoly(Black_MASK, Region_of_Interest_Vertices, Color_of_MASK_Area_to_Ignore)
    ROI_masked_image = cv2.bitwise_and(Color_Gradients_result, Black_MASK)

    if not is_Video:
        Display_Two_Adjacent_Images(ROI_Color_Gredient_result, "Using ROI on the Color & Gradients result", None,
                                ROI_masked_image, "Mask all regions except the ROI", None)

    # Define the Perspective Region of Interest (Source) - WAF
    # Source_Vertix_upper_left  = (600, 440)  # (x , y)
    # Source_Vertix_upper_right = (685, 440)
    # Source_Vertix_lower_right = (1150, 670)
    # Source_Vertix_lower_left  = (200, 670)

    # Define the Perspective Region of Interest (Source)
    Source_Vertix_upper_left = (575, 465)  # (x , y)
    Source_Vertix_upper_right = (710, 465)
    Source_Vertix_lower_right = (1050, 680)
    Source_Vertix_lower_left = (260, 680)

    Source_Vertices = np.float32([Source_Vertix_upper_left, Source_Vertix_upper_right,
                                  Source_Vertix_lower_right, Source_Vertix_lower_left])
    # get a copy of the undistorted image to draw on
    Prespect_ROI_img = np.copy(undistorted_Test_img)

    # Draw a ROI red line with thickness of 5 px
    cv2.line(Prespect_ROI_img, Source_Vertix_upper_left, Source_Vertix_upper_right, (255, 0, 0), 5)
    cv2.line(Prespect_ROI_img, Source_Vertix_upper_right, Source_Vertix_lower_right, (255, 0, 0), 5)
    cv2.line(Prespect_ROI_img, Source_Vertix_lower_right, Source_Vertix_lower_left, (255, 0, 0), 5)
    cv2.line(Prespect_ROI_img, Source_Vertix_lower_left, Source_Vertix_upper_left, (255, 0, 0), 5)

    # Define the Perspective Destination vertices - WAF
    # Prespect_Vertix_upper_left = (150, 0)  # (x , y)
    # Prespect_Vertix_upper_right = (1200, 0)
    # Prespect_Vertix_lower_right = (1200, 720)
    # Prespect_Vertix_lower_left = (150, 720)

    # Define the Perspective Destination vertices
    Prespect_Vertix_upper_left = (450, 0)  # (x , y)
    Prespect_Vertix_upper_right = (830, 0)
    Prespect_Vertix_lower_right = (830, 720)
    Prespect_Vertix_lower_left = (450, 720)

    Prespect_Vertices = np.float32([Prespect_Vertix_upper_left, Prespect_Vertix_upper_right,
                                    Prespect_Vertix_lower_right, Prespect_Vertix_lower_left])

    Warped_masked_image, M, Minv = Warp(ROI_masked_image, Source_Vertices, Prespect_Vertices)

    if not is_Video:
        Display_Two_Adjacent_Images(Prespect_ROI_img, "Prespective Region of Interest", None,
                                Warped_masked_image, "Warped PROI image.", 'gray')

    # the following constant determines the periodicity of resets in terms of the number of frames.
    reset_span = 8

    if ((frame_count % reset_span) == 0) or frame_count < 3:
        left_lane_poly, right_lane_poly, left_lane_x, left_lane_y, right_lane_x, right_lane_y, \
        Lane_Lines_Ploy_img, histogram = Locate_Lane_Lines_Fit_Polynomial(Warped_masked_image)
        if not is_Video:
            Display_Lane_ploy(histogram, Lane_Lines_Ploy_img, left_lane_poly, right_lane_poly)
            # only for images not for videos
            left_lane_poly, right_lane_poly, left_lane_x, left_lane_y, right_lane_x, right_lane_y, Lane_Lines_Ploy_img = \
                Recursive_Search_Lane_Lines(Warped_masked_image, Left_Lane.current_fit, Right_Lane.current_fit)
    else:
        left_lane_poly, right_lane_poly, left_lane_x, left_lane_y, right_lane_x, right_lane_y, Lane_Lines_Ploy_img = \
            Recursive_Search_Lane_Lines(Warped_masked_image, Left_Lane.current_fit, Right_Lane.current_fit)

    if is_Video:
        frame_count = frame_count + 1

    # Calculating the radius of curvature (left & right lanes) in meters
    left_curve_radius, right_curve_radius,  Car_Dist_from_Lane_Center = \
        Measure_Curvature_and_Center_Dist(Lane_Lines_Ploy_img, left_lane_poly, right_lane_poly, left_lane_x,
                                          left_lane_y, right_lane_x, right_lane_y)

    if not is_Video:
        print('Radius of curvature for the lanes in the image: left ', np.int(left_curve_radius), 'm, right ',
          np.int(right_curve_radius), 'm')
        print('Distance from lane center for example: {:.2f}'.format(Car_Dist_from_Lane_Center), 'm')

    Highlighted_Lane_img = HighLight_Lane(undistorted_Test_img, Warped_masked_image,
                                          left_lane_poly, right_lane_poly, Minv)

    # Display the following information on the output images / Video
    font = cv2.FONT_HERSHEY_SIMPLEX
    white_color =(255,255,255)
    Left_lane_curve_text  = "Radius of Left Curvature : {} m".format(np.int(left_curve_radius))
    Right_lane_curve_text = "Radius of Right Curvature: {} m".format(np.int(right_curve_radius))
    Car_Distance_text     = "Distance from Lane Center: {:.2f} m".format(Car_Dist_from_Lane_Center)
    cv2.putText(Highlighted_Lane_img, Left_lane_curve_text, (400, 50), font, 1, white_color, 2)
    cv2.putText(Highlighted_Lane_img, Right_lane_curve_text, (400, 100), font, 1, white_color, 2)
    cv2.putText(Highlighted_Lane_img, Car_Distance_text, (400, 150), font, 1, white_color, 2)

    if not is_Video:
        plt.figure(figsize=(20, 10))
        plt.imshow(Highlighted_Lane_img)
        plt.title('Located Lane Highlighted')
        plt.show()

    return Highlighted_Lane_img
#======================================================================================
#======================= End Process Image Function ===================================
#======================================================================================
# Declare two Line() classes for left & right lanes
Left_Lane = Line()
Right_Lane = Line()
# #======================================================================================
# #        Start Image Testing here
# #======================================================================================
# if not is_Video:
#     for image_file_name in os.listdir(Test_images_path):
#         # read the Test Images
#         Test_img = mpimg.imread(Test_images_path + image_file_name)
#         # Apply the pipeline on the image
#         Image_with_Detected_Lane_Lines = process_image(Test_img)
#         filename, extention = image_file_name.split(".")
#         out_file_name = Output_path + filename + "_out." + extention
#         print('\n', filename + "_out." + extention)
#         cv2.imwrite(out_file_name, cv2.cvtColor(Image_with_Detected_Lane_Lines, cv2.COLOR_BGR2RGB))
#         # plt.imshow(Image_with_Detected_Lane_Lines, interpolation='nearest', aspect='auto')
#         # plt.savefig(out_file_name, dpi=1000)
#
# #=====================================================================================
# # Start Video Testing here
# #=====================================================================================
# if is_Video:
#     video_output1 = Video_path + 'project_video_output.mp4'
#     video_input1 = VideoFileClip(Video_path+'project_video.mp4')#.subclip(12,14)
#     #video_input1 = VideoFileClip(Video_path + 'challenge_video.mp4')
#     processed_video = video_input1.fl_image(process_image_lane)
#     processed_video.write_videofile(video_output1, audio=False)
# #====================================================================================
