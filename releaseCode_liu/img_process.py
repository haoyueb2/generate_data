import numpy as np
import cv2
import random
import xmlWrite
import os
import utils
import math
from PIL import Image,ImageDraw
#from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array

save_to_dir = 'D:/ML/VOC/VOCdevkit/VOC2007/temp'

def random_process(img_path,xml_path,copy_path,change,iter_array,rotation,anchor,area,subfix):
    A = cv2.imread(img_path)


    (h,w,c) = A.shape
    print("the shape is",(w,h,c))
    if change:
        A = utils.add_filter(A)
 #   A = utils.sp_noise(A,0.001)
 #   A = cv2.GaussianBlur(A,ksize=(3,3),sigmaX=1,sigmaY=1)
    ### 添加高斯噪声
    A = utils.GaussianNoise(A,0,9)

    img_path = img_path.split('\\')[-1].split('.')[0]
    xml_path = os.path.join(xml_path,img_path+str('.xml'))

    B,iter_num = utils.release_difference(A,img_path,xml_path,copy_path,iter_array,rotation,anchor,area,subfix)

    return B,iter_num

# def img_process_new(save_to_dir_path,img_path,xml_path,copy_path,origin_path,change,max,isDiff,isOffset):
#     A = cv2.imread(img_path)
#
#     (h, w, c) = A.shape
#     print("the shape is", (w, h, c))
#     if change:
#         A = utils.add_filter(A)
#     #   A = utils.sp_noise(A,0.001)
#     #   A = cv2.GaussianBlur(A,ksize=(3,3),sigmaX=1,sigmaY=1)
#     A = utils.GaussianNoise(A, 0, 9)
#
#     if isDiff:
#         probility = random.random()
#         if probility > 0.05:
#             isDiff = False
#     if not isDiff and isOffset:
#         probility = random.random()
#         if probility > 0.05:
#             isOffset = False
#
#     root_path = img_path
#     img_path = img_path.split('\\')[-1].split('_A.')[0]
#     xml_path = os.path.join(xml_path, img_path + str('.xml'))
#
#     B = utils.process_difference(A, root_path,save_to_dir_path, img_path, xml_path, copy_path,origin_path,max,isDiff,isOffset)
#
#     return B
#
#
#
# def judge_pos(left_top,right_bottom,label):
#     # if label[0] <= left_top[0] <= label[2] and label[1] <= left_top[1] <= label[3]:
#     #     return True
#     # if label[0] <= right_bottom[0] <= label[2] and label[1] <= right_bottom[1] <= label[3]:
#     #     return True
#     # if label[0] <= right_bottom[0] <= label[2] and label[1] <= left_top[1] <= label[3]:
#     #     return True
#     # if label[0] <= left_top[0] <= label[2] and label[1] <= right_bottom[1] <= label[3]:
#     #     return True
#     # return False
#     if right_bottom[0] < label[0] or label[2] < left_top[0]:
#         return False
#     if right_bottom[1] < label[1] or label[3] < left_top[1]:
#         return False
#     return True
#
# def judge_compare(origin_array,change_array):
#     div_array = origin_array-change_array
#     div_array = np.abs(div_array)
#     result = np.sum(div_array)
#     if result > 20:
#         return False
#     return True
#
#
#
# def copy_rectangle(iter,A,B,w,h,c,label,xml_path,img_path):
#     for i in range(iter):
#         m = 0
#         iter_judge = False
#         while 1:
#             if m == 50:
#                 iter_judge = True
#             random_w = random.randint(0, w - 30)
#             random_h = random.randint(0, h - 30)
#             size_w = random.randint(30, w-10)
#             size_h = random.randint(30, h-10)
#             random_w_t = random.randint(0, w - 30)
#             random_h_t = random.randint(0, h - 30)
#             temp_w = max(random_w, random_w_t)
#             temp_h = max(random_h, random_h_t)
#             size_w = w - 1 - temp_w if temp_w + size_w > w - 1 else size_w
#             size_h = h - 1 - temp_h if temp_h + size_h > h - 1 else size_h
#             judge = True
#             for i in range(len(label)):
#                 if judge_pos([random_w_t, random_h_t], [random_w_t + size_w, random_h_t + size_h], label[i]):
#                     judge = False
#                     break
#             if judge_compare(B[random_h_t:random_h_t + size_h, random_w_t:random_w_t + size_w],
#                              A[random_h:random_h + size_h, random_w:random_w + size_w]):
#                 m = m+1
#                 continue
#             if judge:
#                 break
#         if iter_judge:
#             continue
#         label.append([random_w_t, random_h_t, random_w_t + size_w, random_h_t + size_h])
#         print("info area [{},{},{},{}]".format(random_w, random_h, random_w + size_w, random_h + size_h))
#         print("changed area [{},{},{},{}]".format(random_w_t, random_h_t, random_w_t + size_w, random_h_t + size_h))
#         try:
#             B[random_h_t:random_h_t + size_h, random_w_t:random_w_t + size_w] = A[random_h:random_h + size_h,
#                                                                                 random_w:random_w + size_w]
#         #       B = cv2.rectangle(B,(random_w,random_h),(random_w+size_w,random_h+size_h),(255,0,0),2)
#         #       B = cv2.rectangle(B, (random_w_t, random_h_t), (random_w_t + size_w, random_h_t + size_h), (255, 0, 0), 2)
#         except Exception as e:
#             print(e)
#     xmlWrite.convert_xml(label, xml_path, img_path, [w, h, c])
#     return B
#
# def copy_random(iter,A,B,w,h,c,xml_path,img_path):
#     point = [4]
#     point_num = random.choice(point)
#     another_path = 'D:/ML/VOC/VOCdevkit/VOC2007/2'
#     img_list = os.listdir(another_path)
#     w = A.shape[1]
#     h = A.shape[0]
#     print(w, h)
#     C_offset = np.zeros((h, w, 3), dtype=np.uint8)
#     f = C_offset.copy()
#     crops = []
#     area_factor =  0.4
#     m_i = 0
#     # set iter
#     iter = 1
#     while m_i < iter:
#         while 1:
#             while 1:
#                 points = []
#                 C_points = []
#                 max_h = 0
#                 min_h = 999
#                 size_w = w // 3
#                 size_h = h // 3
#                 select_value = [1, 2]
#                 value_factor_w = random.choice(select_value)
#                 value_factor_h = random.choice(select_value)
#
#                 choice = random.choice(select_value)
#                 if choice == 1:
#                     random_w = random.randint(0,30)
#                     random_h = random.randint(10,h-10)
#                     points.append([random_w,random_h])
#                     random_w = random.randint(w-50,w-10)
#                     random_h = random.randint(10,h-10)
#                     points.append([random_w,random_h])
#                 if choice == 2:
#                     random_w = random.randint(10, w-10)
#                     random_h = random.randint(10, 30)
#                     points.append([random_w, random_h])
#                     random_w = random.randint(10, w-10)
#                     random_h = random.randint(h-50, h-10)
#                     points.append([random_w, random_h])
#
#                 for i in range(2):
#                     random_w = random.randint(10,w-10)
#                     random_h = random.randint(10,h-10)
#                     points.append([random_w,random_h])
#                 for i in range(4):
#                     if points[i][1] > max_h:
#                         max_h = points[i][1]
#                     if points[i][1] < min_h:
#                         min_h = points[i][1]
#                 # random_w = size_w * value_factor_w + 10
#                 # random_h = size_h * value_factor_h + 10
#                 # for i in range(point_num):
#                 #     if random_w + size_w + 10 > w:
#                 #         random_w = random_w - size_w
#                 #     if random_h + size_h + 10 > h:
#                 #         random_h = random_h - size_h
#                 #     if random_w - size_w - 10 <= 0:
#                 #         random_w = size_w + 10
#                 #     if random_h - size_h - 10 <= 0:
#                 #         random_h = size_h + 10
#                 #     random_w = random.randint(random_w - size_w, random_w + size_w)
#                 #     random_h = random.randint(random_h - size_h, random_h + size_h)
#                 #     while [random_w, random_h] in point:
#                 #         if random_w + size_w + 10 > w:
#                 #             random_w = random_w - size_w
#                 #         if random_h + size_h + 10 > h:
#                 #             random_h = random_h - size_h
#                 #         if random_w - size_w - 10 <= 0:
#                 #             random_w = size_w + 10
#                 #         if random_h - size_h - 10 <= 0:
#                 #             random_h = size_h + 10
#                 #         random_w = random.randint(random_w - size_w, random_w + size_w)
#                 #         random_h = random.randint(random_h - size_h, random_h + size_h)
#                 #     points.append([random_w, random_h])
#                 #     if random_h > max_h:
#                 #         max_h = random_h
#                 #     if random_h < min_h:
#                 #         min_h = random_h
#                 C_points = points.copy()
#                 points.sort()
#                 w_length = points[point_num - 1][0] - points[0][0]
#                 h_length = max_h - min_h
#                 if w_length * h_length > w * h *area_factor: #or w_length * h_length < w * h * 0.2:
#                     continue
#                 if (points[2][0] - points[1][0] != points[1][0] - points[0][0]) or (
#                         points[2][1] - points[1][1] != points[1][1] - points[0][1]):
#                     maskIm = Image.new('L', (w, h), 0)
#                     polygon = [tuple(x) for x in C_points]
#                     ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
#                     maskIm = np.array(maskIm).reshape(h, w, 1)
#                     value = np.sum(maskIm)
#                     threshold = 0.3
#                     if value < w_length * h_length *threshold:
#                         continue
#                     print(1)
#
#
#
#
#                     break
#
#             # print(C)
#
#             while 1:
#                 copy_x = random.randint(0, w - 10 - w_length)
#                 max_h_u = max(10 + points[0][1] - min_h, h - max_h + points[0][1] - 10)
#                 min_h_u = min(10 + points[0][1] - min_h, h - max_h + points[0][1] - 10)
#                 copy_y = random.randint(min_h_u, max_h_u)
#                 h_top = copy_y + max_h - points[0][1]
#                 h_bottom = copy_y - points[0][1] + min_h
#                 h_c_t = min(h_top, max_h)
#                 h_c_d = max(h_bottom, min_h)
#                 x_right = copy_x + w_length
#                 x_c_l = max(copy_x, points[0][0])
#                 x_c_r = min(x_right, points[-1][0])
#                 judge = False
#                 break
#
#                 # if (x_c_r - x_c_l) * (h_c_t - h_c_d) < w_length * h_length * 0.4:
#                 #     print(2)
#                 #     for i in range(len(crops)):
#                 #         if judge_pos([copy_x, h_bottom], [x_right, h_top], crops[i]):
#                 #             judge = True
#                 #             break
#                 #     break
#             # if not judge:
#             #     break
#             break
#
#         maskIm = np.concatenate([maskIm, maskIm, maskIm], axis=2)
#         # print(maskIm)
#         copy_img = np.zeros((h, w, 3), dtype=np.uint8)
#         img_temp_path = os.path.join(another_path,random.choice(img_list))
#         img_temp = cv2.imread(img_temp_path)
#         img_temp = cv2.resize(img_temp,(w,h))
#         P = np.where(img_temp, img_temp, 1)
#         C = np.where(maskIm, P, 0)
#
#         M = np.where(C)
#
#         x_offset = points[0][0] - copy_x
#         y_offset = points[0][1] - copy_y
#         C_h = M[0] - y_offset
#         C_w = M[1] - x_offset
#         C_offset[C_h, C_w, M[2]] = C[M]
#
#         maskIm = Image.new('L', (w, h), 0)
#         polygon = []
#         for i in range(point_num):
#             x_temp = points[i][0]-x_offset
#             y_temp = points[i][1]-y_offset
#             polygon.append(tuple([x_temp,y_temp]))
#         ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
#         maskIm = np.array(maskIm).reshape(h, w, 1)
#         maskIm = np.concatenate([maskIm, maskIm, maskIm], axis=2)
#         C_A = np.where(maskIm,P,0)
#         if abs(np.sum(C_A-C_offset)) < w_length*h_length*1:
#             continue
#
#
#         final = np.where(C_offset, C_offset, A)
#         np.uint8(final)
#  #       f = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)
#  #       cv2.rectangle(f, (copy_x, h_bottom), (x_right, h_top), (255, 0, 0), 2)
#  #       C_offset = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
#         C_offset = final.copy()
#         #    print((copy_x,h_bottom),(x_right,h_top))
#         crops.append([copy_x, h_bottom, x_right, h_top])
#         xmlWrite.convert_xml(crops, xml_path, img_path, [w, h, c])
#         m_i = m_i+1
#
#     return final
#
#
# def copy_random_rectangle(iter,A,B,w,h,c,xml_path,img_path):
#     point = [4]
#     point_num = random.choice(point)
#     another_path = 'D:/ML/VOC/VOCdevkit/VOC2007/2'
#     img_list = os.listdir(another_path)
#     w = A.shape[1]
#     h = A.shape[0]
#     print(w, h)
#     C_offset = np.zeros((h, w, 3), dtype=np.uint8)
#     f = C_offset.copy()
#     crops = []
#     area_factor =  0.4
#     m_i = 0
#     # set iter
#     iter = 1
#     while m_i < iter:
#         while 1:
#             while 1:
#                 points = []
#                 C_points = []
#                 max_h = 0
#                 min_h = 999
#                 size_w = w // 3
#                 size_h = h // 3
#                 select_value = [1, 2]
#                 value_factor_w = random.choice(select_value)
#                 value_factor_h = random.choice(select_value)
#                 angle = random.uniform(0.15,0.35)
#                 choice = random.choice(select_value)
#               #   if choice == 1:
#               #       random_w = random.randint(10,50)
#               #       random_h = random.randint(10,h-10)
#               #       points.append([random_w,random_h])
#               #       random_w = random.randint(w-50,w-10)
#               #       random_h = random.randint(10,h-10)
#               #       points.append([random_w,random_h])
#               #   if choice == 2:
#               #       random_w = random.randint(10, w-10)
#               #       random_h = random.randint(10, 50)
#               #       points.append([random_w, random_h])
#               #       random_w = random.randint(10, w-10)
#               #       random_h = random.randint(h-50, h-10)
#               #       points.append([random_w, random_h])
#                 random_w = random.randint(10, 50)
#                 random_h = random.randint(40,h-40)
#                 points.append([random_w,random_h])
#                 pro_width = min(h,w-30)
#                 t_width = random.randint((int)(2*pro_width*1.4/3),(int)(pro_width*1.4))
#                 random_w = (int)(random_w+t_width*math.cos(math.pi*angle))
#                 if choice == 1:
#                     random_h = (int)(random_h+t_width*math.sin(math.pi*angle))
#                 elif choice == 2:
#                     random_h = (int)(random_h-t_width*math.sin(math.pi*angle))
#                 points.append([random_w,random_h])
#
#
#                 random_w = random.randint(10,w-10)
#                 random_h = random.randint(10,h-10)
#                 points.append([random_w,random_h])
#                 final_w = points[2][0] + points[1][0] - points[0][0]
#                 final_h = points[2][1] + points[1][1] - points[0][1]
#                 if final_w < 0 or final_w > w-5:
#                     continue
#                 if final_h < 0 or final_h > h-5:
#                     continue
#                 points.append([final_w,final_h])
#
#                 for i in range(len(points)):
#                     if points[i][1] < min_h:
#                         min_h = points[i][1]
#                     if points[i][1] > max_h:
#                         max_h = points[i][1]
#                 if min_h < 0 or max_h > h:
#                     continue
#                 C_points = points.copy()
#                 points.sort()
#                 # random_w = size_w * value_factor_w +
#                 # 10
#                 # random_h = size_h * value_factor_h + 10
#                 # for i in range(point_num):
#                 #     if random_w + size_w + 10 > w:
#                 #         random_w = random_w - size_w
#                 #     if random_h + size_h + 10 > h:
#                 #         random_h = random_h - size_h
#                 #     if random_w - size_w - 10 <= 0:
#                 #         random_w = size_w + 10
#                 #     if random_h - size_h - 10 <= 0:
#                 #         random_h = size_h + 10
#                 #     random_w = random.randint(random_w - size_w, random_w + size_w)
#                 #     random_h = random.randint(random_h - size_h, random_h + size_h)
#                 #     while [random_w, random_h] in point:
#                 #         if random_w + size_w + 10 > w:
#                 #             random_w = random_w - size_w
#                 #         if random_h + size_h + 10 > h:
#                 #             random_h = random_h - size_h
#                 #         if random_w - size_w - 10 <= 0:
#                 #             random_w = size_w + 10
#                 #         if random_h - size_h - 10 <= 0:
#                 #             random_h = size_h + 10
#                 #         random_w = random.randint(random_w - size_w, random_w + size_w)
#                 #         random_h = random.randint(random_h - size_h, random_h + size_h)
#                 #     points.append([random_w, random_h])
#                 #     if random_h > max_h:
#                 #         max_h = random_h
#                 #     if random_h < min_h:
#                 #         min_h = random_h
#
#                 w_length = points[point_num - 1][0] - points[0][0]
#                 h_length = max_h - min_h
#                 # if w_length * h_length > w * h *area_factor: #or w_length * h_length < w * h * 0.2:
#                 #     continue
#                 if (points[2][0] - points[1][0] != points[1][0] - points[0][0]) or (
#                         points[2][1] - points[1][1] != points[1][1] - points[0][1]):
#                     maskIm = Image.new('L', (w, h), 0)
#                     polygon = [tuple(x) for x in C_points]
#                     polygon[2],polygon[3] = polygon[3],polygon[2]
#                     ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
#                     maskIm = np.array(maskIm).reshape(h, w, 1)
#                     value = np.sum(maskIm)
#                     threshold = 0.25
#                     if value < w_length * h_length *threshold:
#                         continue
#                     print(1)
#                     break
#
#             # print(C)
#
#             while 1:
#                 copy_x = random.randint(0, w - 10 - w_length)
#                 max_h_u = max(10 + points[0][1] - min_h, h - max_h + points[0][1] - 10)
#                 min_h_u = min(10 + points[0][1] - min_h, h - max_h + points[0][1] - 10)
#                 copy_y = random.randint(min_h_u, max_h_u)
#                 h_top = copy_y + max_h - points[0][1]
#                 h_bottom = copy_y - points[0][1] + min_h
#                 h_c_t = min(h_top, max_h)
#                 h_c_d = max(h_bottom, min_h)
#                 x_right = copy_x + w_length
#                 x_c_l = max(copy_x, points[0][0])
#                 x_c_r = min(x_right, points[-1][0])
#                 judge = False
#
#                 if (x_c_r - x_c_l) * (h_c_t - h_c_d) < w_length * h_length * 1:
#                     print(2)
#                     for i in range(len(crops)):
#                         if judge_pos([copy_x, h_bottom], [x_right, h_top], crops[i]):
#                             judge = True
#                             break
#                     break
#             if not judge:
#                 break
#
#
#         maskIm = np.concatenate([maskIm, maskIm, maskIm], axis=2)
#         # print(maskIm)
#         copy_img = np.zeros((h, w, 3), dtype=np.uint8)
#         img_temp_path = os.path.join(another_path,random.choice(img_list))
#         img_temp = cv2.imread(img_temp_path)
#         img_temp = cv2.resize(img_temp,(w,h))
#         P = np.where(img_temp, img_temp, 1)
#         C = np.where(maskIm, P, 0)
#
#         M = np.where(C)
#
#         x_offset = points[0][0] - copy_x
#         y_offset = points[0][1] - copy_y
#         C_h = M[0] - y_offset
#         C_w = M[1] - x_offset
#         C_offset[C_h, C_w, M[2]] = C[M]
#
#         maskIm = Image.new('L', (w, h), 0)
#         polygon = []
#         for i in range(point_num):
#             x_temp = points[i][0]-x_offset
#             y_temp = points[i][1]-y_offset
#             polygon.append(tuple([x_temp,y_temp]))
#         ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
#         maskIm = np.array(maskIm).reshape(h, w, 1)
#         maskIm = np.concatenate([maskIm, maskIm, maskIm], axis=2)
#         C_A = np.where(maskIm,P,0)
#         if abs(np.sum(C_A-C_offset)) < w_length*h_length*1:
#             continue
#
#
#         final = np.where(C_offset, C_offset, A)
#         np.uint8(final)
#  #       f = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)
#  #       cv2.rectangle(f, (copy_x, h_bottom), (x_right, h_top), (255, 0, 0), 2)
#  #       C_offset = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
#         C_offset = final.copy()
#         #    print((copy_x,h_bottom),(x_right,h_top))
#         crops.append([copy_x, h_bottom, x_right, h_top])
#         xmlWrite.convert_xml(crops, xml_path, img_path, [w, h, c])
#         m_i = m_i+1
#
#     return final
#
#
# def copy_random_singleColor(iter,A,B,w,h,c,xml_path,img_path):
#     point = [3, 4]
#     point_num = random.choice(point)
#
#     w = A.shape[1]
#     h = A.shape[0]
#     print(w, h)
#     C_offset = np.zeros((h, w, 3), dtype=np.uint8)
#     f = C_offset.copy()
#     crops = []
#     final = A.copy()
#     area_factor =  0.3
#     m_i = 0
#     while m_i < iter:
#         while 1:
#             points = []
#             C_points = []
#             max_h = 0
#             min_h = 999
#             size_w = w // 4
#             size_h = h // 4
#             select_value = [1, 2, 3]
#             value_factor_w = random.choice(select_value)
#             value_factor_h = random.choice(select_value)
#             random_w = size_w * value_factor_w + 10
#             random_h = size_h * value_factor_h + 10
#             for i in range(point_num):
#                 if random_w + size_w + 10 > w:
#                     random_w = random_w - size_w
#                 if random_h + size_h + 10 > h:
#                     random_h = random_h - size_h
#                 if random_w - size_w - 10 <= 0:
#                     random_w = size_w + 10
#                 if random_h - size_h - 10 <= 0:
#                     random_h = size_h + 10
#                 random_w = random.randint(random_w - size_w, random_w + size_w)
#                 random_h = random.randint(random_h - size_h, random_h + size_h)
#                 while [random_w, random_h] in point:
#                     if random_w + size_w + 10 > w:
#                         random_w = random_w - size_w
#                     if random_h + size_h + 10 > h:
#                         random_h = random_h - size_h
#                     if random_w - size_w - 10 <= 0:
#                         random_w = size_w + 10
#                     if random_h - size_h - 10 <= 0:
#                         random_h = size_h + 10
#                     random_w = random.randint(random_w - size_w, random_w + size_w)
#                     random_h = random.randint(random_h - size_h, random_h + size_h)
#                 points.append([random_w, random_h])
#                 if random_h > max_h:
#                     max_h = random_h
#                 if random_h < min_h:
#                     min_h = random_h
#             C_points = points.copy()
#             points.sort()
#             w_length = points[point_num - 1][0] - points[0][0]
#             h_length = max_h - min_h
#             if w_length * h_length > w * h * area_factor or w_length * h_length < w * h * 0.01:
#                 continue
#             if (points[2][0] - points[1][0] != points[1][0] - points[0][0]) or (
#                     points[2][1] - points[1][1] != points[1][1] - points[0][1]):
#                 maskIm = Image.new('L', (w, h), 0)
#                 polygon = [tuple(x) for x in C_points]
#                 ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
#                 maskIm = np.array(maskIm).reshape(h, w, 1)
#                 value = np.sum(maskIm)
#                 threshold = 0.3
#                 if value < w_length * h_length *threshold:
#                     continue
#                 judge = False
#                 for i in range(len(crops)):
#                     if judge_pos([points[0][0], min_h], [points[point_num - 1][0], max_h], crops[i]):
#                         judge = True
#                         break
#                 if judge:
#                     continue
#
#
#                 break
#
#         maskIm = np.concatenate([maskIm, maskIm, maskIm], axis=2)
#         # print(maskIm)
#         copy_img = np.zeros((h, w, 3), dtype=np.uint8)
#         P = np.where(A, A, 1)
# #        C = np.where(maskIm, P, 0)
#         channel_one = np.random.randint(1,255)
#         channel_two = np.random.randint(1, 255)
#         channel_three = np.random.randint(1, 255)
#         maskIm[:,:,0] = np.where(maskIm[:,:,0],channel_one,0)
#         maskIm[:, :, 1] = np.where(maskIm[:, :, 1], channel_two, 0)
#         maskIm[:, :, 2] = np.where(maskIm[:, :, 2], channel_three, 0)
#         final = np.where(maskIm,maskIm,final)
#         np.uint8(final)
#  #
#    #     cv2.rectangle(f, (points[0][0], min_h), (points[point_num-1][1], max_h), (255, 0, 0), 2)
#   #      final = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
#         #C_offset = final.copy()
#         #    print((copy_x,h_bottom),(x_right,h_top))
#         crops.append([points[0][0], min_h, points[point_num-1][0], max_h])
#         m_i = m_i+1
#
#     print(crops)
#
#
#     xmlWrite.convert_xml(crops, xml_path, img_path, [w, h, c])
#     return final

if __name__ == '__main__':
    img_path = 'C:\\Users\\a\\Pictures\\gouqishui.jpg'
    xml_path = 'C:\\Users\\a\\Pictures'
    A = random_process(img_path,xml_path,change=True,shift=True)
    cv2.imshow("111",A)
    cv2.waitKey()