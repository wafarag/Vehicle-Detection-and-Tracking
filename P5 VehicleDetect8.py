################### P5 Vehicle Detection   #####################
#                                                              #
#                    Done by Wael Farag                        #
################################################################

# Import the used Libraries
import os
import cv2
import glob
import time
import pickle
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.ndimage.measurements import label
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from P4_AdvLane3 import *

#===============================================================================
# Some Initializations for parameters:
#===============================================================================

# flag to turn on or off data visualization
# Visualize_ok = True
Visualize_ok = False

# flag to indicate if the training data is already prepared or not
# So you can load the trained data directly
Training_Data_Ready = True
# Training_Data_Ready = False

# flag to indicate if the training is already done or not
# So you can load the trained model directly
Training_Done = True
# Training_Done = False

# Flag to disable any verbosity or display during Video processing
#is_Video = True
is_Video = False

Output_path = "./outputs/TrainResults3/"
Test_images_path = "./test_images/"
Output_images_path = "./output_images/"
Video_path = "./"

# This could be 'RGB', 'HSV', 'HSL', 'LUV', 'YUV', 'YCrCb' or 'LAB'
COLOR_SPACE = 'HSV'

# ============================================================================
# load Training Data
if not Training_Data_Ready or Visualize_ok:

    # vehicles_data = glob.glob('TrainingData/vehicles/**/*.png')
    # non_vehicles_data = glob.glob('TrainingData/non-vehicles/**/*.png')

    # load the raw data from the provided databases of images
    raw_vehicles_data = glob.glob('TrainingData/vehicles/**/*.png')
    raw_non_vehicles_data = glob.glob('TrainingData/non-vehicles/**/*.png')


    # Augment the data by flipping the images (both vehicles and non-vehicles)
    vehicles_data = []
    for raw_image in raw_vehicles_data:
        data_image = cv2.imread(raw_image, cv2.COLOR_BGR2RGB)
        flipped_data_image = cv2.flip(data_image, 1)
        vehicles_data.append(data_image)
        vehicles_data.append(flipped_data_image)

    non_vehicles_data = []
    for raw_image in raw_non_vehicles_data:
        data_image = cv2.imread(raw_image, cv2.COLOR_BGR2RGB)
        flipped_data_image = cv2.flip(data_image, 1)
        non_vehicles_data.append(data_image)
        non_vehicles_data.append(flipped_data_image)

    # Vehicles Data Size    :  8792 x 2 = 17,584
    vehicles_data_length = len(vehicles_data)
    # Non-Vehicles Data Size:  8968 x 2 = 17,936
    non_vehicles_data_length = len(non_vehicles_data)

# ================================================================================
# ==================== Start of Data Visualization ===============================

if Visualize_ok:
    # print lengths of data sets
    print('Vehicles Data Size    : ', vehicles_data_length)
    print('Non-Vehicles Data Size: ', non_vehicles_data_length)

    # fig, axs = plt.subplots(5, 6, figsize=(16, 16))
    fig, axs = plt.subplots(5, 10)
    fig.subplots_adjust(hspace=.3, wspace=.001)
    plt.suptitle('Vehicles Data Visualization', fontsize=16)
    axs = axs.ravel()

    for i in range(0, 50):
        # Display Vehicle images
        img_index = rnd.randint(0, vehicles_data_length)
        image = plt.imread(vehicles_data[img_index])
        axs[i].axis('off')
        axs[i].set_title('# {}'.format(img_index), fontsize=10)
        axs[i].imshow(image)

    plt.show()
    fig, axs = plt.subplots(5, 10)
    fig.subplots_adjust(hspace=.3, wspace=.001)
    plt.suptitle('Non-Vehicles Data Visualization', fontsize=16)
    axs = axs.ravel()

    for i in range(0, 50):
        # Display Non-Vehicle images
        img_index = rnd.randint(0, non_vehicles_data_length)
        image = plt.imread(non_vehicles_data[img_index])
        axs[i].axis('off')
        axs[i].set_title('# {}'.format(img_index), fontsize=10)
        axs[i].imshow(image)

    plt.show()
# ======================= End of Data Visualization ===============================
# =================================================================================
# =================================================================================
# ======================= Define HOG feature extraction function  =================

# Define a function to return HOG features and visualization of these features
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Return both the features and the image outputs if vis==True to visualize the HOG
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise reurn only the features output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

# =========================== End - HOG feature extraction function =================
# ===================================================================================

if Visualize_ok:
    # Just select any sample image from the Vehicles Data and make HOG processing on it
    # Vehicle Image size: (64, 64, 3)
    sample_vehicle_image = cv2.imread(vehicles_data[200], cv2.COLOR_BGR2RGB)
    # sample_vehicle_image = mpimg.imread(vehicles_data[200])
    # print(sample_vehicle_image.shape)
    sample_vehicle_image_gray = cv2.cvtColor(sample_vehicle_image, cv2.COLOR_RGB2GRAY)
    # hog_features, vehicle_hog_img = get_hog_features(sample_vehicle_image[:, :, 2], 9, 8, 2,
    #                                                  vis=True, feature_vec=True)
    hog_features, vehicle_hog_img = get_hog_features(sample_vehicle_image_gray, 9, 8, 2,
                                                     vis=True, feature_vec=True)

    # Just select any sample image from the Non-Vehicles Data and make HOG processing on it
    sample_non_vehicle_image = cv2.imread(non_vehicles_data[200], cv2.COLOR_BGR2RGB)
    # sample_non_vehicle_image = mpimg.imread(non_vehicles_data[200])
    sample_non_vehicle_image_gray = cv2.cvtColor(sample_non_vehicle_image, cv2.COLOR_RGB2GRAY)
    # hog_features, non_vehicle_hog_img = get_hog_features(sample_non_vehicle_image[:, :, 2], 9, 8, 2,
    #                                                      vis=True, feature_vec=True)
    hog_features, non_vehicle_hog_img = get_hog_features(sample_non_vehicle_image_gray, 9, 8, 2,
                                                         vis=True, feature_vec=True)

    # Visualize the resultant HOG Features Images
    # plt.figure(num= 1, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    fig, axs = plt.subplots(2, 2)
    fig.subplots_adjust(hspace=.4, wspace=.1)
    axs= axs.ravel()
    # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    plt.suptitle('HOG Features Visualization', fontsize=14)
    axs[0].axis('off')
    axs[0].imshow(sample_vehicle_image)
    axs[0].set_title('Vehicle Image', fontsize=12)
    axs[1].axis('off')
    axs[1].imshow(vehicle_hog_img, cmap='gray')
    axs[1].set_title('Vechile HOG', fontsize=12)
    axs[2].axis('off')
    axs[2].imshow(sample_non_vehicle_image)
    axs[2].set_title('Non-Vehicle Image', fontsize=12)
    axs[3].axis('off')
    axs[3].imshow(non_vehicle_hog_img, cmap='gray')
    axs[3].set_title('Non-Vechile HOG', fontsize=12)
    fig.tight_layout()
    plt.show()

#=================================================================================
#=================================================================================

# Define a function to compute binned color features
# def bin_spatial(img, size=(32, 32)):
#     # Use cv2.resize().ravel() to create the feature vector
#     spacial_features = cv2.resize(img, size).ravel()
#     # Return the feature vector
#     # print("spacial_features: ", np.array(spacial_features).shape)
#     return spacial_features

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    # print("hist_features: ", np.array(hist_features).shape)
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_color_features(imgs, cspace='RGB', spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = cv2.imread(file, cv2.COLOR_BGR2RGB)
        # image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'LAB':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features)))
    # Return list of feature vectors
    return features
#====================================================================================

# Define a function to extract features from a list of images
def extract_hog_features(imgs, cspace='RGB', orient=9,
                         pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for image in imgs:
        # Read in each one by one
        # image = cv2.imread(file, cv2.COLOR_BGR2RGB)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            elif cspace == 'LAB':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        else:
            feature_image = np.copy(image)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features

# ===================================================================================

if not Training_Data_Ready and not Training_Done:
    print('Data Preparation ...')

    # vehicles_hog_features:  (8792, 5292)
    # No. of hog features in each color = 7 x 7 x 2 x 2 x 9 = 1764
    # No. of total hog features = 1764 x 3 = 5292
    vehicles_hog_features = extract_hog_features(vehicles_data, cspace=COLOR_SPACE, orient=9,
                                                 pix_per_cell=8, cell_per_block=2, hog_channel='ALL')

    # non_vehicles_hog_features:  (8968, 5292)
    # No. of hog features in each color = 7 x 7 x 2 x 2 x 9 = 1764
    # No. of total hog features = 1764 x 3 = 5292
    non_vehicles_hog_features = extract_hog_features(non_vehicles_data, cspace=COLOR_SPACE, orient=9,
                                                     pix_per_cell=8, cell_per_block=2, hog_channel='ALL')

    # vehicles_features:  (8792, 5292)
    vehicles_features = vehicles_hog_features
    # print("vehicles_features: ", np.array(vehicles_features).shape)

    # non_vehicles_features:(8968, 5292)
    non_vehicles_features = non_vehicles_hog_features
    # print("non_vehicles_features: ", np.array(non_vehicles_features).shape)

    # Save the Training Data on file for later use
    # Save in "pickle" file
    Training_Features = {}
    Training_Features["vehicles_features"] = vehicles_features
    Training_Features["non_vehicles_features"] = non_vehicles_features
    pickle.dump(Training_Features, open(Output_path + "training_features.p", "wb"))

# if Training_Data_Ready and not Training_Done:
if Training_Data_Ready:
    # Training_Features = pickle.load(open("training_features.p", "rb"))
    Training_Features = pickle.load(open(Output_path + "training_features.p", "rb"))
    vehicles_features = Training_Features["vehicles_features"]
    non_vehicles_features = Training_Features["non_vehicles_features"]

# =============================================================================================
# ===================  End Data Preparation Section  ==========================================
# =============================================================================================

#======================= Start the training procedure for the Classifier ======================

# print('vehicles_features', np.shape(vehicles_features))
# print('non_vehicles_features', np.shape(non_vehicles_features))

#  Check if training already done before - No need to do it again
if not Training_Done:
    print("Training ...")
    # Create an array stack of feature vectors
    X = np.vstack((vehicles_features, non_vehicles_features)).astype(np.float64)
    print(np.shape(X))

    # Define the labels vector
    y = np.hstack((np.ones(len(vehicles_features)), np.zeros(len(non_vehicles_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

    # Feature vector length: 5292
    print('Feature vector length:', len(X_train[0]))

    # Use a Linear Support Vector Machine Classifier "SVC"
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    # 677.95 Seconds to train SVC...
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    # Test Accuracy of SVC =  0.9778
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    # My SVC predicts:  [ 1.  1.  0.  1.  1.  0.  0.  0.  0.  0.]
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    # For these 10 labels:  [ 1.  1.  0.  1.  1.  0.  0.  0.  0.  0.]
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    # 0.41652 Seconds to predict 10 labels with SVC
    print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')
    # save model
    # save the classifier
    SVC_Trained_Model = svc
    # save with numpy joblib
    _ = joblib.dump(SVC_Trained_Model, Output_path + 'SVM_classifier2.pkl', compress=9)
    # save with Pickle
    with open(Output_path + 'SVM_classifier1.pkl', 'wb') as fid:
        pickle.dump(SVC_Trained_Model, fid)

# load the already trained and saved before Support Vector Machine Classifier
if Training_Done:
    # SVC_Trained_Model = pickle.load(open(Output_path+"SVM_classifier1.pkl", "rb"))
    SVC_Trained_Model = joblib.load(Output_path + "SVM_classifier2.pkl")


# ========================================================================================================
# ===================  End - Training the Classifier Section =============================================

# ============================================================================================================

def convert_color(img, color_space='RGB'):
    if color_space == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if color_space == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if color_space == 'LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if color_space == 'HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if color_space == 'YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    if color_space == 'LAB':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    if color_space == 'RGB':
        return img


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, color_space, xstart, xstop, ystart, ystop, scale, svc, orient, pix_per_cell, cell_per_block, step_size):

    Hit_Boxes = []
    img_tosearch = img[ystart:ystop, xstart:xstop, :]
    ctrans_tosearch = convert_color(img_tosearch, color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = step_size  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Scale features and make a prediction
            test_features = hog_features.reshape(1, -1)
            # plt.plot(test_features[0, :])
            # plt.show()
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            # test_prediction = svc.predict(test_features)

            # Using decision function instead of predict to reduce False Positives
            test_prediction = svc.decision_function(test_features)
            # print(test_prediction)

            # Convert probabilites to Boolean with Confidence score value
            Confidence_score = 0.5
            test_prediction[test_prediction >= Confidence_score] = 1
            test_prediction[test_prediction < Confidence_score]  = 0

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                x1 = xbox_left + xstart
                y1 = ytop_draw + ystart
                x2 = xbox_left + win_draw + xstart
                y2 = ytop_draw + win_draw + ystart
                Hit_Boxes.append([x1, y1, x2, y2])

    return Hit_Boxes


# ===================================================================================


# ======================================================================================================

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):

    global x1, x2, y1, y2

    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        #print(np.shape(nonzero))
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Biult FIR filters for the Boxes vertices
        if is_Video:

            x1[0, car_number] = np.min(nonzerox)
            y1[0, car_number] = np.min(nonzeroy)
            x2[0, car_number] = np.max(nonzerox)
            y2[0, car_number] = np.max(nonzeroy)

            # x1_filtered = np.int((x1[0,car_number] + x1[1,car_number] + x1[2,car_number] + x1[3,car_number]) / 4)
            # y1_filtered = np.int((y1[0,car_number] + y1[1,car_number] + y1[2,car_number] + y1[3,car_number]) / 4)
            # x2_filtered = np.int((x2[0,car_number] + x2[1,car_number] + x2[2,car_number] + x2[3,car_number]) / 4)
            # y2_filtered = np.int((y2[0,car_number] + y2[1,car_number] + y2[2,car_number] + y2[3,car_number]) / 4)

            # x1_filtered = np.int((x1[0, car_number] + x1[1, car_number] + x1[2, car_number] ) / 3)
            # y1_filtered = np.int((y1[0, car_number] + y1[1, car_number] + y1[2, car_number] ) / 3)
            # x2_filtered = np.int((x2[0, car_number] + x2[1, car_number] + x2[2, car_number] ) / 3)
            # y2_filtered = np.int((y2[0, car_number] + y2[1, car_number] + y2[2, car_number] ) / 3)

            x1_filtered = np.int((0.4 * x1[0, car_number] + 0.35 * x1[1, car_number] + 0.25 * x1[2, car_number]))
            y1_filtered = np.int((0.4 * y1[0, car_number] + 0.35 * y1[1, car_number] + 0.25 * y1[2, car_number]))
            x2_filtered = np.int((0.4 * x2[0, car_number] + 0.35 * x2[1, car_number] + 0.25 * x2[2, car_number]))
            y2_filtered = np.int((0.4 * y2[0, car_number] + 0.35 * y2[1, car_number] + 0.25 * y2[2, car_number]))

            for i in range(3,0,-1):
                x1[ i , car_number] = x1[ i-1 , car_number]
                y1[ i , car_number] = y1[ i-1 , car_number]
                x2[ i , car_number] = x2[ i-1 , car_number]
                y2[ i , car_number] = y2[ i-1 , car_number]

        else:
            x1_filtered = np.min(nonzerox)
            y1_filtered = np.min(nonzeroy)
            x2_filtered = np.max(nonzerox)
            y2_filtered = np.max(nonzeroy)

        # Calculate the diagonals of found boxes to represnt sizes
        diag = np.sqrt((y2_filtered - y1_filtered)**2 + (x2_filtered - x1_filtered)**2)

        Rectandle_Diag_Length_Threshold = 90

        if diag > Rectandle_Diag_Length_Threshold and y1_filtered >= 400 and y2_filtered >= 400:
            bbox = ((x1_filtered, y1_filtered), (x2_filtered, y2_filtered))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (255, 0, 0), 6)

        # Define a bounding box based on min/max x and y

    # Return the image
    return img


# ======================================================================================================



# ======================================================================================================

frame_counter = 0

def process_image(Test_img):


    global heat_hist
    global frame_counter, previous_img

    Highlighted_Lane_img = process_image_lane(Test_img)

    sampling_base = 2

    if (frame_counter % sampling_base) == 0 or not is_Video:

        # This is the heat thresold to filter out "False Positives"
        HEAT_THRESHOLD = 1.5

        test_img = np.copy(Highlighted_Lane_img)

        All_Hit_Boxes = []

        # Start the Car Boxes Search -----
        orient = 9  # HOG orientations
        pix_per_cell = 8  # HOG pixels per cell
        cell_per_block = 2  # HOG cells per block
        xstart = 800
        xstop = 1280
        ystart = 400
        ystop = 650
        step_size = 1
        Scale_Step = 0.25
        Scale_Multiplier_Start = 6
        Scale_Multiplier_End = 10
        for i in range(Scale_Multiplier_Start, Scale_Multiplier_End + 1):
            scale = i * Scale_Step
            Hit_Boxes = find_cars(test_img, COLOR_SPACE, xstart, xstop, ystart, ystop, scale,
                                  SVC_Trained_Model, orient, pix_per_cell, cell_per_block, step_size)
            if All_Hit_Boxes == [] and Hit_Boxes != []:
                All_Hit_Boxes = Hit_Boxes
            elif All_Hit_Boxes != [] and Hit_Boxes != []:
                All_Hit_Boxes = np.concatenate((All_Hit_Boxes, Hit_Boxes), 0)

        # Next Car Box Search with different ROI
        xstart = 700
        xstop = 1100
        ystart = 400
        ystop = 500
        step_size = 1
        Scale_Step = 0.2
        Scale_Multiplier_Start = 4
        Scale_Multiplier_End = 10
        for i in range(Scale_Multiplier_Start, Scale_Multiplier_End + 1):
            scale = i * Scale_Step
            Hit_Boxes = find_cars(test_img, COLOR_SPACE, xstart, xstop, ystart, ystop, scale,
                                  SVC_Trained_Model, orient, pix_per_cell, cell_per_block, step_size)
            if All_Hit_Boxes == [] and Hit_Boxes != []:
                All_Hit_Boxes = Hit_Boxes
            elif All_Hit_Boxes != [] and Hit_Boxes != []:
                All_Hit_Boxes = np.concatenate((All_Hit_Boxes, Hit_Boxes), 0)

        # Next Car Box Search with different ROI
        xstart = 1150
        xstop = 1280
        ystart = 400
        ystop = 580
        step_size = 1
        Scale_Step = 0.25
        Scale_Multiplier_Start = 6
        Scale_Multiplier_End = 12
        for i in range(Scale_Multiplier_Start, Scale_Multiplier_End + 1):
            scale = i * Scale_Step
            Hit_Boxes = find_cars(test_img, COLOR_SPACE, xstart, xstop, ystart, ystop, scale,
                                  SVC_Trained_Model, orient, pix_per_cell, cell_per_block,step_size)
            if All_Hit_Boxes == [] and Hit_Boxes != []:
                All_Hit_Boxes = Hit_Boxes
            elif All_Hit_Boxes != [] and Hit_Boxes != []:
                All_Hit_Boxes = np.concatenate((All_Hit_Boxes, Hit_Boxes), 0)

        # Next Car Box Search with different ROI
        scale  = 1.5
        xstart = 0
        xstop  = 330
        ystart = 400
        ystop  = 650
        step_size = 2
        Hit_Boxes = find_cars(test_img, COLOR_SPACE, xstart, xstop, ystart, ystop, scale,
                              SVC_Trained_Model, orient, pix_per_cell, cell_per_block, step_size)
        if All_Hit_Boxes == [] and Hit_Boxes != []:
            All_Hit_Boxes = Hit_Boxes
        elif All_Hit_Boxes != [] and Hit_Boxes != []:
            All_Hit_Boxes = np.concatenate((All_Hit_Boxes, Hit_Boxes), 0)


        # print(np.shape(All_Hit_Boxes))
        # print(scale)

        if All_Hit_Boxes != []:
            for x1, y1, x2, y2 in All_Hit_Boxes:
                cv2.rectangle(test_img, (x1, y1), (x2, y2), (255, 0, 0), 6)

        if Visualize_ok and not is_Video:
            plt.imshow(test_img)
            plt.show()

        # cv2.imshow('Output Image',test_img)
        # cv2.waitKey(0)

        Hit_Boxes_List = []
        for x1, y1, x2, y2 in All_Hit_Boxes:
            Hit_Boxes_List.append(((x1, y1), (x2, y2)))

        heat = np.zeros_like(test_img[:, :, 0]).astype(np.float)

        # Add heat to each box in box list
        heat = add_heat(heat, Hit_Boxes_List)

        # FIR filtering for the heatmaps
        if is_Video:
            heat_hist[0] = heat
            filtered_heat = (0.3 * heat_hist[0] + 0.4 * heat_hist[1] + 0.4 * heat_hist[2] +
                             0.3 * heat_hist[3] + 0.1 * heat_hist[4])
            for i in range(4, 0, -1):
                heat_hist[i] = heat_hist[i - 1]
        else:
            filtered_heat = heat

        # Apply threshold to help remove false positives
        heat = apply_threshold(filtered_heat, HEAT_THRESHOLD)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        #print(np.shape(labels[0]))
        draw_img = draw_labeled_bboxes(np.copy(Highlighted_Lane_img), labels)

        previous_img = draw_img

        if Visualize_ok and not is_Video:
            fig = plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(draw_img)
            plt.title('Car Positions')
            plt.subplot(1, 2, 2)
            plt.imshow(heatmap, cmap='hot')
            plt.title('Heat Map')
            fig.tight_layout()
            plt.show()

        frame_counter = frame_counter + 1
        Car_Detection_image = draw_img
    else:
        frame_counter = frame_counter + 1
        Car_Detection_image = previous_img


    return Car_Detection_image


# ========================================================================================================
if not is_Video:
    for image_file_name in os.listdir(Test_images_path):
        # read the Test Images
        Test_img = mpimg.imread(Test_images_path + image_file_name)
        # Apply the pipeline on the image
        Image_with_Detected_Cars = process_image(Test_img)
        filename, extention = image_file_name.split(".")
        out_file_name = Output_images_path + filename + "_out." + extention
        print('\n', filename + "_out." + extention)
        cv2.imwrite(out_file_name, cv2.cvtColor(Image_with_Detected_Cars, cv2.COLOR_BGR2RGB))
        # plt.imshow(Image_with_Detected_Lane_Lines, interpolation='nearest', aspect='auto')
        # plt.savefig(out_file_name, dpi=1000)

# ================================================================================================================
if is_Video:

    # Initializations for the FIR heatmaps
    heat_hist = np.zeros((5,720,1280)).astype(np.float)

    # Initializations for the FIR heatmaps
    x1 = np.zeros((4, 20))
    y1 = np.zeros((4, 20))
    x2 = np.zeros((4, 20))
    y2 = np.zeros((4, 20))

    video_output1 = Video_path + 'project_video_output.mp4'
    video_input1 = VideoFileClip(Video_path + 'project_video.mp4')  # .subclip(12,14)

    # video_output1 = Video_path + 'test_video_output.mp4'
    # video_input1 = VideoFileClip(Video_path + 'test_video.mp4')  # .subclip(12,14)

    processed_video = video_input1.fl_image(process_image)
    processed_video.write_videofile(video_output1, audio=False)