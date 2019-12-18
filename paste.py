import PIL
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
from pascal_voc_writer import Writer
from skimage.util import random_noise


def RandomPaste(origin, img, bndbox, overlapping):
    # Resize the pasted image
    width, height = origin.size
    propotion = random.uniform(0.1, 0.7)
    img_width = int(propotion * width)
    img_height = int(propotion * height)
    img = img.resize((img_width, img_height), Image.ANTIALIAS)
    
    # Rotate the pasted image
    rotate_angle = random.randint(0, 360)
    img = img.rotate(rotate_angle, expand=True)
    
    # Crop extra edges
    maxsize = (width / 2, height / 2)
    img.thumbnail(maxsize, Image.ANTIALIAS)
    imageSize = img.size
    imageComponents = img.split()
    rgbImage = Image.new("RGB", imageSize, (0,0,0))
    rgbImage.paste(img, mask=imageComponents[3])
    croppedBox = rgbImage.getbbox()
    img = img.crop(croppedBox)
    
    # Paste image
    r,g,b,a = img.split()
    img_width, img_height = img.size
    img_x = int(random.uniform(0, 1) * (width - img_width))
    img_y = int(random.uniform(0, 1) * (height - img_height))
    newbox = [img_x, img_y, img_width, img_height]
    
    if overlapping:
        origin.paste(img, (img_x, img_y), a)
        bndbox = combine_boxes(bndbox, newbox)
    elif not is_overlapping(bndbox, newbox):
        origin.paste(img, (img_x, img_y), a)
        bndbox.append(newbox)
    else:
        return False
    
    return True


def union(a, b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0] + a[2], b[0] + b[2]) - x
    h = max(a[1] + a[3], b[1] + b[3]) - y
    return [x, y, w, h]


def intersection(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0: 
        return ()
    return [x, y, w, h]


def combine_boxes(boxes, new):
    flag = False
    for i in range(len(boxes) - 1, -1, -1):
        if intersection(boxes[i], new):
            flag = True
            newbox = union(boxes[i], new)
            boxes.remove(boxes[i])
            combine_boxes(boxes, newbox)
            
    if not flag:
        boxes.append(new)
        
    return boxes


def is_overlapping(boxes, new):
    for b in boxes:
        if intersection(b, new):
            return True
    return False


def add_noise(img):
    mode = ['gaussian','localvar','poisson','salt','pepper','s&p','speckle']
    img_arr = np.asarray(img)
    mode_index = random.uniform(0,8)
    mode_index = int(mode_index)
    #暂定高斯
    mode_index = 0
    if mode[mode_index] == 'gaussian' or mode[mode_index] =='speckle':
        devia = random.uniform(0,0.1)
        noise_img = random_noise(img_arr, mode='gaussian', var=devia**2)
    else:
        noise_img = random_noise(img_arr, mode = mode[mode_index])
        
    noise_img = (255*noise_img).astype(np.uint8)
    img = Image.fromarray(noise_img)
    return img


def convert_temp(img):
    kelvin_table = {
    1000: (255,56,0),
    1500: (255,109,0),
    2000: (255,137,18),
    2500: (255,161,72),
    3000: (255,180,107),
    3500: (255,196,137),
    4000: (255,209,163),
    4500: (255,219,186),
    5000: (255,228,206),
    5500: (255,236,224),
    6000: (255,243,239),
    6500: (255,249,253),
    7000: (245,243,255),
    7500: (235,238,255),
    8000: (227,233,255),
    8500: (220,229,255),
    9000: (214,225,255),
    9500: (208,222,255),
    10000: (204,219,255)}

    temp = random.choice(list(kelvin_table.keys()))
    r, g, b = kelvin_table[temp]
    matrix = ( r / 255.0, 0.0, 0.0, 0.0,
               0.0, g / 255.0, 0.0, 0.0,
               0.0, 0.0, b / 255.0, 0.0 )
    return img.convert('RGB', matrix)


def generatePictures(num_pair, count=0, mode=0, small_diff=False, word=False, num_obj=None, noise_type=None, deviation=False, overlapping=False, display=False):
    '''
    Parameters
    ----------
    - num_pair: 生成的图片对总数
    - count: 图片开始的编号(从0开始)
    - mode: 生成图片的模式
        - 0: 有贴图
        - 1: 没有贴图（可认为一致，不画框）
        - 2: 两张图完全不同（整个框起来）
    - small_diff: 存在微小差异（不被框出来的那种）
    - word: 在左上角或右下角随机生成日期等文字
    - num_obj: 每张图上贴的物体数，默认为1～5，实际上可能没有那么多，详见overlapping
    - noise_type: 噪声类型，默认高斯噪声
    - deviation: 图片偏移，默认不偏移
    - overlapping: 贴图是否重叠，默认不重叠。不重叠的情况下，有些贴图不会贴上去，也就是贴图总数可能不同于num_obj
    - display: 是否展示贴图和框的效果，用于测试
    '''
    
    bg_path = "./img/background"
    obj_path = "./img/object"
    images_path = "./data/images"
    xml_path = "./data/XML"
    
    for i in range(num_pair):
        count = count + 1
        if mode is 0 and deviation is True:
            identifier = "dev_" + str(count)
        elif mode is 0 and overlapping is True:
            identifier = "lap_" + str(count)
        elif mode is 1:
            identifier = "same_" + str(count)
        elif mode is 2:
            identifier = "diff_" + str(count)
        else:
            identifier = str(count)

        bndbox = []
        bg_name = random.choice(os.listdir(bg_path))
        background = Image.open(os.path.join(bg_path, bg_name))
        background.save(os.path.join(images_path, identifier + '.jpg'))

        # 正常情况
        if mode is 0:
            if num_obj is None:
                num_obj = random.randint(1, 5);

            for i in range(num_obj):
                obj = Image.open(os.path.join(obj_path, random.choice(os.listdir(obj_path))))
                RandomPaste(background, obj, bndbox, overlapping)

        # 两张图不存在差异
        if mode is 1:
            pass

        # 两张图完全不同
        if mode is 2:
            dir_list = os.listdir(bg_path)
            dir_list.remove(bg_name)
            imname = random.choice(dir_list)
            cols, rows = background.size
            background = Image.open(os.path.join(bg_path, imname))
            opencvImage = cv2.cvtColor(np.asarray(background),cv2.COLOR_RGB2BGR)
            background = cv2.resize(opencvImage, (cols, rows), interpolation=cv2.INTER_CUBIC)
            background = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
            bndbox.append([0, 0, cols, rows])
            deviation=False

        # Add noise
        background = add_noise(background)
        
        # Add deviation
        dx = dy = 0
        if deviation is True:
            width, height = background.size[:2]
            dx = int(random.uniform(-1 * width / 30, width / 30))
            dy = int(random.uniform(-1 * height / 30, height / 30))

            opencvImage = cv2.cvtColor(np.asarray(background),cv2.COLOR_RGB2BGR)
            
            T = np.float32([[1, 0, dx], [0, 1, dy]])
            img_translation = cv2.warpAffine(opencvImage, T, (width, height), borderMode=cv2.BORDER_REFLECT)
            
            if display is True:
                plt.figure(figsize=(10, 20))
                plt.subplot(1,2,1)
                plt.imshow(background)
                plt.subplot(1,2,2)
                background = Image.fromarray(cv2.cvtColor(img_translation, cv2.COLOR_BGR2RGB))
                plt.imshow(background) 
                plt.show()
                print((dx, dy))
            else:
                background = Image.fromarray(cv2.cvtColor(img_translation, cv2.COLOR_BGR2RGB))
            
        # Adjust Bounding Boxes
        for b in bndbox:
            b[0] = min(b[0], b[0] + dx)
            b[1] = min(b[1], b[1] + dy)
            b[2] = b[2] + abs(dx)
            b[3] = b[3] + abs(dy)

        # Adjust Color Temperature
        background = convert_temp(background)
        
        
        # Draw boxes in picture
        if display is not False:
            draw = ImageDraw.Draw(background)
            for b in bndbox:
                draw.rectangle((b[0], b[1], b[0]+ b[2], b[1] + b[3]), outline='red')
            plt.figure(figsize=(10, 20))
            plt.imshow(background)
        
        # Save Images
        background.save(os.path.join(images_path, identifier + '_A.jpg'))
        
        # Write Boxes to XML
        width, height = background.size
        writer = Writer(identifier + ".jpg", width, height)
        for b in bndbox :
            writer.addObject('True', b[0], b[1], b[0] + b[2], b[1] + b[3])
        writer.save(os.path.join(xml_path, identifier + ".xml"))


if __name__ == "__main__":
    # 完全一致的十对图片
    generatePictures(10, 0, 1)
    # 完全不一致的十对图片
    generatePictures(10, 0, 2)
