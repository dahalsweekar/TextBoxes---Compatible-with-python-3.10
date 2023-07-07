import os
import math
from PIL import Image
import matplotlib.pyplot as plt

test_file = './evaluation/test_list.txt'
img_dir = './dataset/test_dataset/'
gt_dir = './evaluation/gt/'
dt_dir = './evaluation/dt/'

print('Calculating Metrics:')


# Find the intersection and union of dt and gt
def calculate_metrics(gt, dt, range_rate):
    hit_recall = 0
    hit_precision = 0

    flag = [0 for t in range(len(dt))]
    if len(dt) != 0:
        for j in range(0, len(gt)):
            for k in range(0, len(dt)):
                arr1 = dt[k].split(',')
                arr2 = gt[j].split(',')

                polygon1_x = [arr1[0], arr1[2]]
                polygon1_y = [arr1[1], arr1[3]]

                polygon2_x = [arr2[0], arr2[2]]
                polygon2_y = [arr2[1], arr2[3]]

                intersect_area = 0
                union_area = 1
                set_range = range_rate

                if (int(polygon2_x[0]) - set_range <= int(polygon1_x[0]) and (int(polygon1_x[0]) <=
                                                                              int(polygon2_x[0]) + set_range)):
                    if (int(polygon2_x[1]) - set_range <= int(polygon1_x[1]) and (int(polygon1_x[1]) <= int(
                            polygon2_x[1]) + set_range)):
                        if (int(polygon2_y[0]) - set_range <= int(polygon1_y[0]) and (int(polygon1_y[0]) <=
                                                                                      int(polygon2_y[
                                                                                              0]) + set_range)):
                            if (int(polygon2_y[1]) - set_range <= int(polygon1_y[1]) and (int(polygon1_y[1]) <= int(
                                    polygon2_y[1]) + set_range)):

                                # print('Polygon1_x')
                                # print(polygon1_x)
                                # print('Polygon1_y')
                                # print(polygon1_y)
                                # print('Polygon2_x')
                                # print(polygon2_x)
                                # print('Polygon2_y')
                                # print(polygon2_y)

                                length_gt = abs(int(polygon2_x[1]) - int(polygon2_x[0]))
                                width_gt = abs(int(polygon2_y[1]) - int(polygon2_y[0]))

                                length_dt = abs(int(polygon1_x[1]) - int(polygon1_x[0]))
                                width_dt = abs(int(polygon1_y[1]) - int(polygon1_y[0]))

                                area_gt = length_gt * width_gt
                                area_dt = length_dt * width_dt
                                diff_area = abs(area_dt - area_gt)
                                xt = abs(int(polygon1_x[0]) - int(polygon2_x[0]))
                                yt = abs((int(polygon1_y[1]) - int(polygon2_y[1])))
                                if int(polygon1_x[0]) <= int(polygon2_x[0]):
                                    csx = abs(int(polygon1_x[1]) - xt)
                                else:
                                    csx = abs(int(polygon2_x[1]) - xt)

                                if int(polygon1_y[0]) <= int(polygon2_y[0]):
                                    csy = abs(int(polygon1_y[1]) - yt)
                                else:
                                    csy = abs(int(polygon2_y[1]) - yt)
                                intersect_area = csx * csy
                                union_area = abs(diff_area + intersect_area)
                if union_area == 0:
                    IoU = 0
                else:
                    IoU = intersect_area / union_area
                if IoU > threshold:
                    flag[k] = math.ceil(IoU)
    for item in flag:
        if item > 0:
            hit_recall = hit_recall + 1
            hit_precision = hit_precision + 1
    return hit_recall, hit_precision


# Calculating the range under which the detected texts are located for varied sized images
def calc_range(ratio):
    x_mean, y_mean = (1000, 1000)
    image = Image.open(image_path)
    image_size = image.size
    x = image_size[0]
    y = image_size[1]
    total_pixels = x * y
    approx_range = (1 - ratio) * 100
    range_rate = total_pixels / (x_mean * y_mean) * (approx_range * approx_range)
    range_rate = math.sqrt(range_rate)
    return range_rate


test_list = open(test_file, 'r')
lines = test_list.readlines()
images = []
for line in lines:
    images.append(line.strip())
test_list.close()

threshold = 0.95

nImg = len(images)
print(f'Total Images: {nImg}')
cum_hit_recall = 0
cum_hit_precision = 0
total_gt = 0
total_dt = 0

for i in images:
    dt = []
    gt = []
    hit_recall = 0
    hit_precision = 0
    name = i.split('.')[0]
    gt_name = name + '.txt'
    dt_name = 'vgg_' + name + '.txt'
    image_path = os.path.join(img_dir, i)
    shift_rate = calc_range(ratio=0.75)

    # Adding groundtruths texts
    gt_path = os.path.join(gt_dir, gt_name)
    file_gt = open(gt_path, 'r')
    gt_lines = file_gt.readlines()
    for gt_line in gt_lines:
        gt_line = gt_line.strip()
        gt.append(gt_line[:-7])
    file_gt.close()

    # Adding detected texts
    dt_path = os.path.join(dt_dir, dt_name)
    file_dt = open(dt_path, 'r')
    dt_lines = file_dt.readlines()
    for dt_line in dt_lines:
        dt_line = dt_line.strip()
        dt.append(dt_line)
    file_dt.close()

    hit_recall, hit_precision = calculate_metrics(gt, dt, shift_rate)
    cum_hit_recall = cum_hit_recall + hit_recall
    cum_hit_precision = cum_hit_precision + hit_precision
    total_gt = total_gt + len(gt)
    total_dt = total_dt + len(dt)

# Calculate Recall, Precision, F1-Score
recall = cum_hit_recall / total_gt
precision = cum_hit_precision / total_dt
f_measure = (2 * recall * precision) / (recall + precision)
print("Results:")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"F1_score: {f_measure}")
