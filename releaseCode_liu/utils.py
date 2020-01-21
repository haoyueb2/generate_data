import cv2
import numpy as np
import random
import math
import os
from PIL import Image,ImageDraw,ImageFilter
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
import xmlWrite


def add_filter(A):
    C = [10,10,11,11,12,12,13,13]
    result = random.choice(C)
    shape = A.shape
    A[:,:,0] = np.sqrt(A[:,:,0]) * result
    return A

def sp_noise(image,prob):
    '''
    添加椒盐噪声
    prob:噪声比例
    '''
    output = np.zeros(image.shape,np.uint8)
    print(output.shape)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = [0,0,0]
            elif rdn > thres:
                output[i][j] = [255,255,255]
            else:
                output[i][j] = image[i][j]
    print(output.shape)
    return output


def GaussianNoise(img,mu,sigma):
    image = np.array(img / 255, dtype=float)
    noise = np.random.normal(mu, sigma / 255, image.shape)
    out = image + noise

    low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out


def release_difference(A,img_path,xml_path,copy_path,iter_array,rotation,anchor,area,subfix,mode = 1,margin_blur = True):
    '''
    Parameters
    - mode: 生成图片的模式
        - 0: 矩形贴图
        - 1: 多边形贴图
    - margin_blur: 边缘模糊
    '''
    ### 最终输出的图片矩阵
    final = A
    ### 迭代数，因为是随机寻找可裁剪的图片，所以有的时候会陷入死循环，用它来做一个限制
    iter_num = 0
    ### 旋转元素个数
    ro_len = len(rotation)
    ### anchor元素个数
    anchor_len = len(anchor)
    ### 面积元素个数
    area_len = len(area)
    img_list = os.listdir(copy_path)
    h,w,c = A.shape

    crops = []  # save the points to prevent overlap

    for i in range(len(iter_array)):
        ### 所选中的旋转角度的索引
        sele_rotation = 0

        ### 所选中的长宽比例的索引
        sele_anchor = 0

        ### 所选中的面积大小的索引
        sele_area = 0

        ### num的值 应该是 “生成元素”的实际索引(现在的索引因为被shuffle过后打乱了,根据实际索引，可以推出它的rotation
        ### anchor以及area的选择
        num = iter_array[i]


        temp = num // (anchor_len * area_len)

        sele_rotation = temp

        num = num - temp*(anchor_len*area_len)

        temp = num // (area_len)
        sele_anchor = temp

        num = num - temp * area_len
        sele_area = num

        # 如果 旋转的索引 为 1 或者 3 即为 45°或者 135°，我们对其做一个随机 ±30°的旋转偏移
        # 如果为0°，则什么也不做
        if sele_rotation == 1 or sele_rotation == 3:
            use_rotation = rotation[sele_rotation] + random.uniform(-math.pi/6,math.pi/6)
        else:
            use_rotation = rotation[sele_rotation]

        # 保持anchor
        use_anchor = anchor[sele_anchor]

        # 面积也会做一个偏移
        if sele_area == 0:
            use_area = random.uniform(0.005,area[sele_area])
        else:
            use_area = random.uniform(area[sele_area-1],area[sele_area])


        print("rotation",use_rotation)
        print("anchor",use_anchor)
        print("area",use_area)

        # 因为有了占比面积以及长宽比，所以我们能够求出来我们所需要的差异矩形的高和宽。
        # w*h为原图(即要被生成差异的图)的宽和高
        assume_h = math.floor(math.sqrt(w * h * use_area / use_anchor))
        assume_w = (int)(use_anchor*assume_h)
        print("assume_h h",assume_h,h)
        print("assume_w w",assume_w,w)

        # 旋转之后的宽和高
        assume_w_t = max(math.cos(sele_rotation)*assume_w,math.sin(sele_rotation)*assume_h)
        assume_h_t = max(math.sin(sele_rotation)*assume_w,math.cos(sele_rotation)*assume_h)

        # 用来对差异进行面积下降(因为有时候图片放不下这个差异)
        down_factor = 1

        # 如果旋转之后差异的高比原图高或者差异的宽比原图宽
        if assume_h_t > h:
            down_factor = assume_h_t / (h-30)
        if assume_w_t > w:
            if (assume_w/(w-20)) > down_factor:
                down_factor = assume_w/(w-30)
        # 降级处理
        if not (down_factor == 1):
            assume_w = assume_w // down_factor
            assume_h = assume_h // down_factor

        judge = False
        over_judge = 0
        while 1:
            points = []
            over_judge = over_judge+1
            # 如果循环1000次都没有找到合适的图片，那么进行降级处理
            if over_judge > 1000:
                if i == 0:
                    assume_w = assume_w // 1.1
                    assume_h = assume_h // 1.1
                    over_judge = 0
                    print("down!!")
                else:
                    judge = True
                    break

            # 随机选择要裁剪的x和y(左上角的点)
            random_x = random.randint(10,w-10)
            random_y = random.randint(10,h-10)
            min_y = random_y
            max_y = random_y
            points.append([random_x,random_y])


            # 通过选定的x和y，确定剩下的三个点的x和y
            sec_x = random_x + math.floor(math.cos(use_rotation)*assume_w)
            sec_y = random_y - math.floor(math.sin(use_rotation)*assume_w)
            points.append([sec_x, sec_y])
            sub_w = sec_x - random_x
            sub_h = sec_y - random_y

            thi_x = random_x + math.floor(math.sin(use_rotation)*assume_h)
            thi_y = random_y + math.floor(math.cos(use_rotation)*assume_h)
            points.append([thi_x, thi_y])

            fou_x = thi_x + sub_w
            fou_y = thi_y + sub_h
            points.append([fou_x,fou_y])

            judge_loop = False
            # 判断是否可以被原图所容纳
            for m in range(len(points)):
                # 如果超出了边界
                if points[m][0] < 0 or points[m][0] >= w or points[m][1] < 0 or points[m][1] >=h:
                    judge_loop = True
                    break
                if points[m][1] < min_y:
                    min_y = points[m][1]
                if points[m][1] > max_y:
                    max_y = points[m][1]
            if judge_loop:
                continue
            c_points = points.copy()
            points.sort()
            # 判断该裁剪区域是否和已经被裁剪的重合(这里，我应该默认裁剪区域的坐标同时也是被贴到原图上的坐标)
            # 即判断如果这个图贴到原图上会不会和其他已经被贴上的图的框有部分重合
            for m in range(len(crops)):
                if judge_pos([points[0][0],min_y],[points[-1][0],max_y],crops[m]):
                    judge_loop = True
                    break
            if not judge_loop:
                break
        if not judge:
            crops.append([points[0][0],min_y,points[-1][0],max_y])
            # 生成一个mask，面积大小和原图相同
            maskIm = Image.new('L', (w, h), 0)

            # 向mask上要裁剪的区域填充1，其他为0
            if mode == 0:
                polygon = [tuple(x) for x in c_points]
                # 转换一下点的坐标(应该是第二个点和第三个点换一下，这个的顺序和我们生成时添加的顺序有点差别)
                polygon[2], polygon[3] = polygon[3], polygon[2]
            elif mode == 1:
                width = points[-1][0] - points[0][0]
                height = max_y - min_y
                # radius = min(width, height) * random.uniform(0.15, 0.4)
                x_radius = width*0.5
                y_radius = height*0.5
                center_x = (points[0][0] + points[-1][0])/2
                center_y = (min_y + max_y)/2
                spikeyness = random.uniform(0, 0.15)
                irregular = random.uniform(0.4, 0.7)
                # 全0生成标准多边形,n是多边形边数
                # spikeyness = 0
                # irregular = 0
                # n = random.randint(6,12)
                n = 10
                polygon = generatePolygon(center_x, center_y, x_radius,y_radius, irregular, spikeyness, n)



            ImageDraw.Draw(maskIm).polygon(polygon, outline=255, fill=255)

            # 轻度边缘模糊，肉眼不易观察，调大可以看出
            if margin_blur is True:
                maskIm = maskIm.filter(ImageFilter.GaussianBlur(random.uniform(0.8, 1.1)))
            # 随机读取一张图去裁剪差异贴图 贴到原图上
            img_temp_path = os.path.join(copy_path, random.choice(img_list))
            img_temp = cv2.imread(img_temp_path)
            img_temp = cv2.resize(img_temp, (w, h))


            # maskIm = np.array(maskIm).reshape(h, w, 1)
            # # 将其concat成3通道
            # maskIm = np.concatenate([maskIm, maskIm, maskIm], axis=2)
            #
            #
            # # 利用np.where 将mask贴到原图上
            # # P这里的处理是为了防止P矩阵中有0
            # # C的处理时，如果maskIm的当前元素为1，则赋予P的相同位置的像素值，否则则置0，这就将要裁剪的部分从图中抠了出来
            # P = np.where(img_temp, img_temp, 1)
            # C = np.where(maskIm, P, 0)
            # # 抠出来之后，就可以将其贴在原图上
            # final = np.where(C,C, final)
            # np.uint8(final)

            # 注释掉原来的，改由PIL实现粘贴以实现边缘模糊
            pil_final = Image.fromarray(cv2.cvtColor(final,cv2.COLOR_BGR2RGB))
            pil_img_tmp = Image.fromarray(cv2.cvtColor(img_temp,cv2.COLOR_BGR2RGB))
            pil_final.paste(pil_img_tmp,maskIm)

            final = cv2.cvtColor(np.array(pil_final), cv2.COLOR_RGB2BGR)
            iter_num = iter_num+1

            # 多边形的框由于spikyness偏移有些不准，校正一下
            bounding_box = maskIm.getbbox()
            crops.pop()
            crops.append(bounding_box)

        else:
            break
    xmlWrite.convert_xml(crops, xml_path, img_path, [w, h, c],subfix)

    return final,iter_num


def process_difference(A,root_path,save_to_dir,img_path,xml_path,copy_path,origin_path,max,isDiff,isOffset):

    A_h,A_w,A_c = A.shape
    final = A
    crops = []
    if isDiff:
        origin_list = os.listdir(origin_path)
        while 1:
            select_pic = random.choice(origin_list)
            if select_pic.find('_A') != -1 and select_pic.split('_A.')[0] == img_path:
                continue
            B = cv2.imread(origin_path+"/"+select_pic)
            B = cv2.resize(B,(A_w,A_h))
            crops.append([0,0,A_w,A_h])
            xmlWrite.convert_xml(crops, xml_path, img_path, [A_w, A_h, A_c])
            return B

    copy_list = os.listdir(copy_path)

    if isOffset:
        final = do_a_shift(final,save_to_dir)

    for i in range(max):
        C = cv2.imread(copy_path+"/"+random.choice(copy_list))
        for j in range(10):
            h,w,c = C.shape
            down_rate = np.random.uniform(0.6,1.1)
            # h_target = (int)(down_rate*h)
            # w_target = (int)(down_rate*w)
            C_copy = cv2.resize(C,dsize=None,fx=down_rate,fy=down_rate)
            angle = np.random.randint(0,360)
            h,w,c = C_copy.shape
            center = (w//2,h//2)
            M = cv2.getRotationMatrix2D(center,angle,1)
            radians = math.radians(angle)
            sin_value = math.sin(radians)
            cos_value = math.cos(radians)
            target_w = int((h*abs(sin_value))+(w*abs(cos_value)))
            target_h = int((h*abs(cos_value))+(w*abs(sin_value)))
            M[0, 2] += ((target_w/2)-center[0])
            M[1, 2] += ((target_h/2)-center[1])
            Q = cv2.warpAffine(C_copy,M,(target_w,target_h),borderValue=(0,0,0))
            find = False
            for k in range(10000):
                h_s = np.random.randint(0,A_h)
                w_s = np.random.randint(0,A_w)

                if w_s + target_w > A_w or h_s + target_h > A_h:
                    continue
                judge_loop = False
                for m in range(len(crops)):
                    if judge_pos([w_s,h_s],[w_s+target_w,h_s+target_h],crops[m]):
                        judge_loop = True
                        break
                if not judge_loop:
                    find = True
                    break
            if find:
                break
        if not find:
            print("not found")
            continue
        crops.append([w_s,h_s,w_s+target_w,h_s+target_h])
        maskIm = np.zeros((A_h,A_w,3),np.uint8)
        maskIm[h_s:h_s+target_h,w_s:w_s+target_w,:] = Q
        final = np.where(maskIm,maskIm,final)
    xmlWrite.convert_xml(crops, xml_path, img_path, [A_w, A_h, A_c])
    return final
        # maskIm = Image.new('L', (A_w, A_h), 0)
        # polygon = [[w_s,h_s],[w_s+target_w,h_s],[w_s,h_s+target_h],[w_s+target_w,h_s+target_h]]
        # ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
        # maskIm = np.array(maskIm).reshape(A_h, A_w, 1)
        # maskIm = np.concatenate([maskIm, maskIm, maskIm], axis=2)


def judge_pos(left_top,right_bottom,label):

    if right_bottom[0] < label[0] or label[2] < left_top[0]:
        return False
    if right_bottom[1] < label[1] or label[3] < left_top[1]:
        return False
    return True


def do_a_shift(A,save_to_dir):
    width_shift = np.random.uniform(0,0.01)
    height_shift = np.random.uniform(0,0.01)
    data_gen = ImageDataGenerator(width_shift_range=width_shift,height_shift_range=height_shift,fill_mode='nearest')
    A = A[:,:,::-1]
    A = A.reshape((1,)+A.shape)

    for batch in data_gen.flow(A,batch_size=1,save_to_dir=save_to_dir,save_prefix='kk',save_format='jpg'):
        break
    file = os.listdir(save_to_dir)
    A = cv2.imread(os.path.join(save_to_dir,file[0]))
    os.remove(os.path.join(save_to_dir,file[0]))
    return A

#多边形相关
def generatePolygon( ctrX, ctrY, x_radius,y_radius, irregularity, spikeyness, numVerts ) :
    #注释不能顶到头？
    '''
    Start with the centre of the polygon at ctrX, ctrY, 
    then creates the polygon by sampling points on a circle around the centre. 
    Randon noise is added by varying the angular spacing between sequential points,
    and by varying the radial distance of each point from the centre.

    Params:
    ctrX, ctrY - coordinates of the "centre" of the polygon
    aveRadius - in px, the average radius of this polygon, this roughly controls how large the polygon is, really only useful for order of magnitude.
    irregularity - [0,1] indicating how much variance there is in the angular spacing of vertices. [0,1] will map to [0, 2pi/numberOfVerts]
    spikeyness - [0,1] indicating how much variance there is in each vertex from the circle of radius aveRadius. [0,1] will map to [0, aveRadius]
    numVerts - self-explanatory

    Returns a list of vertices, in CCW order.
    '''

    irregularity = clip( irregularity, 0,1 ) * 2*math.pi / numVerts
    spikeyness_x = clip( spikeyness, 0,1 ) * x_radius
    spikeyness_y = clip( spikeyness, 0,1 ) * y_radius
    # generate n angle steps
    angleSteps = []
    lower = (2*math.pi / numVerts) - irregularity
    upper = (2*math.pi / numVerts) + irregularity
    sum = 0
    for i in range(numVerts) :
        tmp = random.uniform(lower, upper)
        angleSteps.append( tmp )
        sum = sum + tmp

    # normalize the steps so that point 0 and point n+1 are the same
    k = sum / (2*math.pi)
    for i in range(numVerts) :
        angleSteps[i] = angleSteps[i] / k

    # now generate the points
    points = []
    # angle = random.uniform(0, 2*math.pi)
    angle = 0
    for i in range(numVerts) :
        a_i = clip( random.gauss(x_radius, spikeyness_x), 0, 2*x_radius )
        b_i = clip( random.gauss(y_radius, spikeyness_y), 0, 2*y_radius )
        #椭圆的参数方程
        x = ctrX + a_i*math.cos(angle)
        y = ctrY + b_i*math.sin(angle)
        points.append( (int(x),int(y)) )

        angle = angle + angleSteps[i]

    return points

def clip(x, min, max) :
    if(min > max ):  
        return x    
    elif( x < min ):  
        return min
    elif( x > max ):  
        return max
    else:     
        return x