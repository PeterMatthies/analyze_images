__author__ = 'petr'

import cv2
import numpy as np
import os
import argparse
import time
import matplotlib.pyplot as plt
import re


# helper function to get specific color channel in grayscale
def get_grayscale_channel(path_to_image, color):
    image = cv2.imread(path_to_image)
    if color == 'red':
        return image, cv2.split(image)[2]
    elif color == 'blue':
        return image, cv2.split(image)[0]
    else:
        return image, cv2.split(image)[1]


# function to find beads using both red and blue images
def find_beads(path_to_red_img, path_to_blue_img, radius):

    # loading images color & grayscale
    red_img, red_ch = get_grayscale_channel(path_to_red_img, 'red')
    blue_img, blue_ch = get_grayscale_channel(path_to_blue_img, 'blue')

    # summing
    sum_ch = cv2.add(blue_ch, red_ch)
    sum_color = cv2.add(blue_img, red_img)

    # finding beads via contours applying different threshold with each step
    beads_com_list, beads_centers = [], []
    for i in range(10, 255, 1):
        ret, thresh = cv2.threshold(sum_ch, i, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            # checking if the contour area is close to the assumed (pi*r^2) bead area
            if np.pi*radius*radius-10 < cv2.contourArea(cnt) < np.pi*radius*radius+10:
                # calculating center of mass
                moments = cv2.moments(cnt)
                center_of_mass_y = int(moments['m01'] / moments['m00'])
                center_of_mass_x = int(moments['m10'] / moments['m00'])
                # check that the center is not at the border of image
                if sum_color.shape[0] - (radius+5) > center_of_mass_y > radius+5 and \
                                        sum_color.shape[1] > center_of_mass_x > radius+5:
                    # drawing found contours
                    cv2.drawContours(sum_color, [cnt], 0, (0, 255, 0), 1)
                    # check for possible duplicates
                    if (center_of_mass_x, center_of_mass_y) not in beads_com_list:
                        beads_com_list.append((center_of_mass_x, center_of_mass_y))
    for i in range(len(beads_com_list)):
        for j in range(i, len(beads_com_list)):
            x = beads_com_list[j][0] - beads_com_list[i][0]
            y = beads_com_list[j][1] - beads_com_list[i][1]
            dist = x*x + y*y
            if dist <= radius*radius+1:
                av_cent_x = int((beads_com_list[j][0] + beads_com_list[i][0])/2)
                av_cent_y = int((beads_com_list[j][1] + beads_com_list[i][1])/2)
                if (av_cent_x, av_cent_y) not in beads_centers:
                    beads_centers.append((av_cent_x, av_cent_y))
            elif beads_com_list[i] not in beads_centers:
                beads_centers.append(beads_com_list[i])
            elif beads_com_list[j] not in beads_centers:
                beads_centers.append(beads_com_list[j])
    #cv2.imshow('contours', sum_color)
    cv2.imwrite('./output/latest_data/2015-07-02_Beads_PE_12samples_output/found_contours_'+path_to_red_img[-29:-9]+'.png', sum_color)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #print beads_com_list
    #print beads_centers
    print len(beads_centers), 'centers found'

    return sum_color, beads_centers


# function to calculate intensities of each bead, background and other things
# like average intensities and intensity at center of mass
def calculate_intensities(image, beads_coords, radius, bg_tresh):
    bg_img = np.copy(image)
    #print beads_coords
    obj_ints_list, peak_ints, avg_ints = [], [], []
    for center in beads_coords:
        # Build mask
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(mask, center, radius, (255, 255, 255), -1, 8, 0)

        # Apply mask (using bitwise & operator)
        result_array = image & mask

        bg_img = bg_img - result_array

        # Crop/center result (assuming center is of the form (x, y))
        result_array = result_array[center[1] - radius:center[1] + radius,
                                    center[0] - radius:center[0] + radius]
        #cv2.imshow('selected circle', result_array)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        # calculating sum object intensity, average obj intensity, center of mass intensity
        obj_int = np.sum(result_array)
        #print obj_int
        avg_int = obj_int/float((len(result_array) * len(result_array)))
        obj_ints_list.append(obj_int)
        avg_ints.append(avg_int)
        peak_ints.append(image[center[1], center[0]])
    # calculating average bg intensity and rms
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    bg_int = np.argmax(hist)
    bg_img = bg_img.flatten()
    rms_list = [abs(int(i)*int(i) - bg_int*bg_int) for i in bg_img if i < bg_tresh]
    bg_rms = sum(rms_list)/float(len(rms_list))
    print 'average bg int', bg_int, '||', 'average bg int rms', bg_rms
    bg_int_list = [bg_int for n in obj_ints_list]
    bg_rms_list = [bg_rms for n in obj_ints_list]
    #cv2.imshow('bg', bg_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return obj_ints_list, peak_ints, avg_ints, bg_int_list, bg_rms_list


start = time.time()

# starting the command line parser
parser = argparse.ArgumentParser(description='script calculating centers of mass '
                                             'of objects and their intensities')
parser.add_argument('--input', action='store', help='specify input directory name', default='./images/')
parser.add_argument('-r', '--radius', action='store', help='specify  radius of the object', default=6)
parser.add_argument('--output', action='store', help='specify output directory name', default='./output/')
parser.add_argument('--thresh', '-t', action='store', help='specify threshold value '
                                                           'for background intensity rms calculation', default=256)
args = parser.parse_args()

# getting input parameters
path = str(args.input)
output_location = str(args.output)
input_radius = int(args.radius)
bg_t = int(args.thresh)
print 'input directory:', path
print 'output directory:', output_location
print 'selected bead radius:', input_radius
print 'bg thresh for rms calc:', bg_t

# getting image names & their sample groups
image_names = []
for root, dirs, files in os.walk(path):
    for d in dirs:
        if not d == 'MetaData':
            for name in os.listdir(path+d):
                if name.endswith('.tif') and name[-8:-6] == 'ch':
                    image_names.append(d+'/'+name)


# grouping them by corresponding channel based on the naming scheme
red_images = [name for name in image_names if name[-8:-4] == 'ch02']
blue_images = [name for name in image_names if name[-8:-4] == 'ch01']
green_images = [name for name in image_names if name[-8:-4] == 'ch00']
samples = [name.split('/')[0] for name in green_images]

# removing all output files of previous program launches
for s in samples:
    current_file = output_location+str(s)+'.txt'
    if os.path.isfile(current_file):
        os.remove(current_file)

# making a beautiful header for output data
column_names = '\t'.join(('object_intensity', 'cm_intensity', 'avg_intensity', 'bg_int', 'bg_rms'))
header = '\t'.join(('center', 'of mass', 'green', '', '', '', '', 'blue', '', '', '', '', 'red', '', '', '', '')) + '\n'
header += 'center_x' + '\t' + 'center_y' + '\t' + column_names + '\t' + column_names + '\t' + column_names + '\n'
#print header


#____________________________________________________________________________________________#
# processing
for r_im, b_im, g_im, s in zip(red_images, blue_images, green_images, samples):
    path_r = path + r_im
    path_g = path + g_im
    path_b = path + b_im
    print path_b
    print path_r
    print path_g
    # finding centers of mass for each bead on image
    im, sum_centers = find_beads(path_r, path_b, input_radius)

    # loading images color & grayscale
    red_image, red_channel = get_grayscale_channel(path_r, 'red')
    blue_image, blue_channel = get_grayscale_channel(path_b, 'blue')
    green_image, green_channel = get_grayscale_channel(path_g, 'green')

    # calculating needed intensities
    obj_ints_red, peaks_red, avg_ints_red, \
        bg_int_red, bg_rms_red = calculate_intensities(red_channel, sum_centers, input_radius, bg_t)
    obj_ints_blue, peaks_blue, avg_ints_blue, \
        bg_int_blue, bg_rms_blue = calculate_intensities(blue_channel, sum_centers, input_radius, bg_t)
    obj_ints_green, peaks_green, avg_ints_green, \
        bg_int_green, bg_rms_green = calculate_intensities(green_channel, sum_centers, input_radius, bg_t)

    # writing calculated intensities to output file based on sample number
    if not os.path.isfile(output_location+str(s)+'.txt'):
        print '_' * 80
        print 'creating new output file for ', s
        with open(output_location+str(s)+'.txt', 'w') as f:   # writing data
            f.write(header)
            for (bead_center,
                 o_i_g, p_i_g, a_i_g, b_i_g, b_r_g,
                 o_i_b, p_i_b, a_i_b, b_i_b, b_r_b,
                 o_i_r, p_i_r, a_i_r, b_i_r, b_r_r) \
                in zip(sum_centers,
                       obj_ints_green, peaks_green, avg_ints_green, bg_int_green, bg_rms_green,
                       obj_ints_blue, peaks_blue, avg_ints_blue, bg_int_blue, bg_rms_blue,
                       obj_ints_red, peaks_red, avg_ints_red, bg_int_red, bg_rms_red):
                f.write('\t'.join((str(bead_center[0]), str(bead_center[1]),
                                   str(o_i_g), str(p_i_g), str(a_i_g), str(b_i_g), str(b_r_g),
                                   str(o_i_b), str(p_i_b), str(a_i_b), str(b_i_b), str(b_r_b),
                                   str(o_i_r), str(p_i_r), str(a_i_r), str(b_i_r), str(b_r_r))) + '\n')
    else:
        print 'appending information for ', s
        with open(output_location+str(s)+'.txt', 'a') as f:   # writing data
            for (bead_center,
                 o_i_g, p_i_g, a_i_g, b_i_g, b_r_g,
                 o_i_b, p_i_b, a_i_b, b_i_b, b_r_b,
                 o_i_r, p_i_r, a_i_r, b_i_r, b_r_r) \
                in zip(sum_centers,
                       obj_ints_green, peaks_green, avg_ints_green, bg_int_green, bg_rms_green,
                       obj_ints_blue, peaks_blue, avg_ints_blue, bg_int_blue, bg_rms_blue,
                       obj_ints_red, peaks_red, avg_ints_red, bg_int_red, bg_rms_red):
                f.write('\t'.join((str(bead_center[0]), str(bead_center[1]),
                                   str(o_i_g), str(p_i_g), str(a_i_g), str(b_i_g), str(b_r_g),
                                   str(o_i_b), str(p_i_b), str(a_i_b), str(b_i_b), str(b_r_b),
                                   str(o_i_r), str(p_i_r), str(a_i_r), str(b_i_r), str(b_r_r))) + '\n')

end = time.time()
print 'seconds elapsed:', end - start


