import os
import numpy as np
import img_process
import cv2
import random
from PIL import Image,ImageDraw,ImageFont
import time
import utils
import math

# rotation  旋转角度
# anchor_ratio 1 => 1:1  2=>1:2
# area 0.05 0.25 0.5 [small medium large]
# word 是否四角添加时间
def imgs_rename(imgs_path,xml_path,copy_path,w_h_ratio,s_ratio,rotation=[0,math.pi/4,math.pi/2,3*math.pi/4],
                anchor_ratio=[1,2,3,5,7],area=[0.05,0.25,0.5],word = True):
    imgs_labels_name = np.array(os.listdir(imgs_path))
    # 从 000001开始
    i = 1
    # 随机选择是原图还是有差异的图进行色差变换
    choice_list = [0,0,0,0,1,1,1,2,2,2,2]  # random pick the image to change light level
    img_number = len(os.listdir(imgs_path))
    sequence = []

    assert len(w_h_ratio) == len(anchor_ratio)
    assert len(s_ratio) == len(area)
    ################################
    ######### 进行序列生成##########
    ################################

    ###len_w_h代表了长宽占比，比如[3 3 3 2 2],代表了anchor_ratio中[1 2 3 5 7]中每个元素的生成比例为了 3:3:3:2:2
    ###同理, s_ratio代表了面积占比，比如[4 2 1]，代表了area中[0.05 0.25 0.5]的每个元素的生成比例为 4:2:1


    ## 长宽占比列表 ##
    len_w_h = len(w_h_ratio)
    ## 面积占比列表 ##
    len_s = len(s_ratio)


    w_h_ratio = np.array(w_h_ratio).reshape(len_w_h,1)
    s_ratio = np.array(s_ratio).reshape(1,len_s)

    ## 长宽有5中选择，面积有3种选择，因此总共有15种选择
    ratio = w_h_ratio * s_ratio

    row,column = ratio.shape
    ### 进行生成统计计数
    statistic = np.zeros(len(anchor_ratio)*len(area))
    for i in range(row):
        for j in range(column):
            for p in range(ratio[i][j]):
                sequence.append(i*column+j)

    temp_sequence = sequence

    ## 将旋转角度也加入生成序列中
    ## 上述生成的temp_sequence可以只视作不做旋转的生成序列
    ## 如果添加上旋转，则需要扩充4倍
    for i in range(len(rotation)):
        if i == 0:
            continue
        temp = map(lambda x:x+i*column*row,temp_sequence)
        sequence = sequence + list(temp)

    sequence_num = len(sequence)
    ### sequence_num 代表我们的“生成选择”的个数[如果是默认的话则是 4*5*3]
    ### img_num 代表我们要进行处理的原图的个数
    ### 这里假设每张图有3个差异(实际会不相同),然后将sequnce扩充到和原图一个数量级
    up_factor = img_number*3/sequence_num
    sequence = sequence * (int)(up_factor)
    sequence = np.array(sequence)
    ### 做一个shuffle来打乱
    np.random.shuffle(sequence)
    sequence = list(sequence)
    ################################
    iter = 0
    i = 1
    for img_label_name in imgs_labels_name:
        subfix = "."  + img_label_name.split('.')[-1]
        result = random.choice(choice_list)
        #### 选择哪张图片来进行色差变化
        if result == 0:
            print(i,"origin changed")
        elif result == 2:
            print(i,"copy changed")

        # 修改图片名称
        img_old_name = os.path.join(os.path.abspath(imgs_path), img_label_name)
        # 类别+图片编号    format(str(i),'0>3s') 填充对齐
        img_new_name = os.path.join(os.path.abspath(imgs_path), 'i00' + format(str(i), '0>4s') + subfix)
        os.rename(img_old_name, img_new_name)
        # 生成错位图片
        method = result == 2
        ### sequence[iter:iter+5] 每次输入5个 “生成选择”元素，有可能全部会被用来生成，但有可能不行。
        ### img_result 所返回的处理过后的图片矩阵
        ### process_num 所使用的 生成元素 的个数
        img_result,process_num = img_process.random_process(img_new_name,xml_path,copy_path,method,sequence[iter:iter+5],rotation,anchor_ratio,area,subfix)


        if word is True:
            pil_img_before_process = Image.open(img_new_name)
            width,height = pil_img_before_process.size
            #设置位置，颜色，字体
            word_loc_list = [(width - 250, height - 30), (3, height - 30), (width - 250, 3), (3, 3)]
            word_loc = random.choice(word_loc_list)
            color_list = ['rgb(255, 255, 255)', 'rgb(0, 0, 0)']
            color = random.choice(color_list)
            font = ImageFont.truetype('arial.ttf', 20)

            #对处理前的图片写时间
            draw_text = ImageDraw.Draw(pil_img_before_process)
            message = time.asctime(time.localtime(time.time()))
            draw_text.text(word_loc, message, fill=color, font=font)
            os.remove(img_new_name)
            pil_img_before_process.save(img_new_name)

            #对处理后的图片写时间
            pil_img_result = Image.fromarray(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
            draw_text = ImageDraw.Draw(pil_img_result)
            message = time.asctime( time.localtime(time.time()+111169) )
            draw_text.text(word_loc,message,fill=color,font = font)
            img_result = cv2.cvtColor(np.array(pil_img_result), cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(os.path.abspath(imgs_path), 'i00' + format(str(i), '0>4s') + "_A"+subfix),img_result)
        if result == 0:
            A = cv2.imread(img_new_name)
            A = utils.add_filter(A)
            os.remove(img_new_name)
            cv2.imwrite(img_new_name,A)
        i = i + 1
        for j in range(process_num):
            num = sequence[iter+j]
            temp = num // (len(w_h_ratio)*len(area))
            num = num - temp*(len(w_h_ratio)*len(area))

            temp = num // (len(area))
            num = num-temp*len(area)

            statistic[temp*len(area)+num] = statistic[temp*len(area)+num] + 1

        iter = iter+process_num

    print(statistic)

if __name__ == '__main__':
    # 要生成差异的一个地址，要放入生成差异图片的原图
    root = "D:/ML/VOC/VOCdevkit/VOC2007/kk"
    # XML的输出路径
    xml_path = 'D:/ML/VOC/VOCdevkit/VOC2007/XML'
    # 放入差异的来源图片
    copy_path = 'D:/ML/VOC/VOCdevkit/VOC2007/Test'
    # w_h_ratio anchor的数量比例
    # s_ratio 面积的数量比例
    imgs_rename(root,xml_path,copy_path,w_h_ratio=[3,3,3,2,2],s_ratio=[4,2,1])
