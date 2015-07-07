__author__ = 'petr'

import cv2
import numpy as np
import os
import argparse
import time
import matplotlib.pyplot as plt
import re


def find_objects_com(path_to_image, color):

    img = cv2.imread(path_to_image, cv2.IMREAD_COLOR)
    blue, green, red = cv2.split(img)
    proc_img = 0
    if color == 'red':
        proc_img = red
    else:
        proc_img = blue

    #cv2.imshow('proc_img', proc_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    proc_img_inv = 255 - proc_img

    ret, thresh = cv2.threshold(proc_img, 15, 255, cv2.THRESH_BINARY)  # +cv2.THRESH_OTSU)

    cv2.imshow('thresh_bin+otsu', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #fg = cv2.erode(thresh, None, iterations=2)
    #bgt = cv2.dilate(thresh, None, iterations=3)
    #ret, bg = cv2.threshold(bgt, 1, 128, 1)
    #marker = cv2.add(fg, bg)
    #cv2.imshow('marker', marker)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #marker32 = np.int32(marker)
    #cv2.watershed(img, marker32)
    #m = cv2.convertScaleAbs(marker32)
    #ret, thresh = cv2.threshold(m, 128, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('tresh', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    objects_com_list = []
    for h, cnt in enumerate(contours):
        #mask1 = np.zeros(image.shape, np.uint8)
        #hist = cv2.calcHist([img], [0], mask1, [256], [0, 256])
        #intensities = [i*hist[i][0] for i in range(len(hist))]
        #objects_ints.append(sum(intensities))
        center_of_mass_x, center_of_mass_y = 0, 0
        #print cv2.contourArea(cnt)
        if cv2.contourArea(cnt) > 25:
            moments1 = cv2.moments(cnt)
            center_of_mass_y = int(moments1['m01'] / moments1['m00'])
            center_of_mass_x = int(moments1['m10'] / moments1['m00'])
            #print center_of_mass_x, center_of_mass_y
        if 405 > center_of_mass_y > 7 and 405 > center_of_mass_x > 7:
            objects_com_list.append((center_of_mass_x, center_of_mass_y))
    return proc_img, objects_com_list


def sel_circ_calc_int(image, centers_coords, radius):

    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    #print hist
    bg_int = np.argmax(hist)
    #plt.hist(proc_img.ravel(),256,[0,256])
    #plt.show()
    obj_ints_list, peak_ints, avg_ints = [], [], []
    for center in centers_coords:
        # Build mask
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(mask, center, radius, (255, 255, 255), -1, 8, 0)

        # Apply mask (using bitwise & operator)
        result_array = image & mask

        # Crop/center result (assuming center is of the form (x, y))
        result_array = result_array[center[1] - radius:center[1] + radius,
                                    center[0] - radius:center[0] + radius]
        #cv2.imshow('selected circle', result_array)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        obj_int = np.sum(result_array)
        avg_int = obj_int/(len(result_array) * len(result_array)) # proverit
        obj_ints_list.append(obj_int)
        avg_ints.append(avg_int)
        #print image[center], image[center[1], center[0]]
        peak_ints.append(image[center[1], center[0]])
    return obj_ints_list, peak_ints, avg_ints, bg_int


start = time.time()
parser = argparse.ArgumentParser(description='script calculating centers of mass '
                                             'of objects and their intensities')
parser.add_argument('--input', action='store', help='specify input directory name', default='./images/')
parser.add_argument('-r', '--radius', action='store', help='specify  radius of the object', default=5)
parser.add_argument('--output', action='store', help='specify output directory name', default='./output/')

args = parser.parse_args()

path = str(args.input)

output_location = str(args.output)
input_radius = args.radius


image_names = os.listdir(path)
#print image_names
print 'input directoty:', path
print 'output directory:', output_location

column_names = '\t'.join(('object_intensity', 'cm_intensity', 'avg_intensity'))

header = '\t'.join(('center', 'of mass', 'green', '', '', 'blue', '', '', 'red', '', '')) + '\n'
header += 'center_x' + '\t' + 'center_y' + '\t' + column_names + '\t' + column_names + '\t' + column_names + '\n'

#print header
number = '3'

red_images = [name for name in image_names if name[-6:-4] == '02']
blue_images = [name for name in image_names if name[-6:-4] == '01']
green_images = [name for name in image_names if name[-6:-4] == '00']
samples = [re.match('^sample(\d+).*', name).group(1) for name in green_images]
print samples


for r_im, b_im, g_im, s in zip(red_images, blue_images, green_images, samples):
    b_b, cm_blue = find_objects_com(path+b_im, 'blue')  # blue
    r_r, cm_red = find_objects_com(path+r_im, 'red')   # red
    sum_centers = []
    for blue_center in cm_blue:
        for red_center in cm_red:
            x = blue_center[0] - red_center[0]
            y = blue_center[1] - red_center[1]
            dist = x*x + y*y
            #print dist
            if dist <= 26:
                av_cent_x = int((blue_center[0] + red_center[0])/2)
                av_cent_y = int((blue_center[1] + red_center[1])/2)
                sum_centers.append((av_cent_x, av_cent_y))
            elif red_center not in sum_centers:
                sum_centers.append(red_center)
            elif blue_center not in sum_centers:
                sum_centers.append(blue_center)

    obj_ints_red, peaks_red, avg_ints_red, bg_int_red = sel_circ_calc_int(r_r, sum_centers, input_radius)
    obj_ints_blue, peaks_blue, avg_ints_blue, bg_int_blue = sel_circ_calc_int(b_b, sum_centers, input_radius)

    green_img = cv2.imread(path+g_im, cv2.IMREAD_COLOR)  # red
    b, g_g, r = cv2.split(green_img)
    obj_ints_green, peaks_green, avg_ints_green, bg_int_green = sel_circ_calc_int(g_g, sum_centers, input_radius)
    if not os.path.isfile(output_location+'sample'+str(s)+'.txt'):
        print 'creating new'
        with open(output_location+'sample'+str(s)+'.txt', 'w') as f:   # writing data
            f.write(header)
            for (center, o_i_g, p_i_g, a_i_g,
                 o_i_b, p_i_b, a_i_b,
                 o_i_r, p_i_r, a_i_r) in zip(sum_centers,
                                             obj_ints_green, peaks_green, avg_ints_green,
                                             obj_ints_blue, peaks_blue, avg_ints_blue,
                                             obj_ints_red, peaks_red, avg_ints_red):
                f.write('\t'.join((str(center[0]), str(center[1]),
                                   str(o_i_g), str(p_i_g), str(a_i_g),
                                   str(o_i_b), str(p_i_b), str(a_i_b),
                                   str(o_i_r), str(p_i_r), str(a_i_r))) + '\n')
    else:
        print 'appending information'
        with open(output_location+'sample'+str(s)+'.txt', 'a') as f:   # writing data
            for (center, o_i_g, p_i_g, a_i_g,
                 o_i_b, p_i_b, a_i_b,
                 o_i_r, p_i_r, a_i_r) in zip(sum_centers,
                                             obj_ints_green, peaks_green, avg_ints_green,
                                             obj_ints_blue, peaks_blue, avg_ints_blue,
                                             obj_ints_red, peaks_red, avg_ints_red):
                f.write('\t'.join((str(center[0]), str(center[1]),
                                   str(o_i_g), str(p_i_g), str(a_i_g),
                                   str(o_i_b), str(p_i_b), str(a_i_b),
                                   str(o_i_r), str(p_i_r), str(a_i_r))) + '\n')
end = time.time()

print 'seconds elapsed:', end - start

#image_3 = '0 (1).jpg'
#image_2 = '0.jpg'
#image_1 = 'Sample7.lif_Image003_ch00.tif'
#image_4 = 'summ.jpg'
#image_1 = './images/sample3.lif_Image020_ch00.tif'
#image_2 = './images/sample3.lif_Image020_ch01.tif'
#image_3 = './images/sample3.lif_Image020_ch02.tif'
#img4 = cv2.imread(image_4, cv2.IMREAD_COLOR) # green
#b4, g4, r4 = cv2.split(img4)



