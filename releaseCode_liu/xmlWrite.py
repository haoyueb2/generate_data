import os
import numpy as np
import codecs
import cv2


##   label [[x_min,y_min,x_max,y_max],[....],[......]]
def convert_xml(label,xml_path,img_name,img_size,subfix):
    xml = codecs.open(xml_path, 'w+', encoding='utf-8')
    xml.write('<annotation>\n')
    xml.write('\t<folder>' + 'VOC2007' + '</folder>\n')
    xml.write('\t<filename>' + img_name + subfix + '</filename>\n')
    xml.write('\t<source>\n')
    xml.write('\t\t<database>The VOC 2007 Database</database>\n')
    xml.write('\t\t<annotation>Pascal VOC2007</annotation>\n')
    xml.write('\t\t<image>flickr</image>\n')
    xml.write('\t\t<flickrid>NULL</flickrid>\n')
    xml.write('\t</source>\n')
    xml.write('\t<owner>\n')
    xml.write('\t\t<flickrid>NULL</flickrid>\n')
    xml.write('\t\t<name>faster</name>\n')
    xml.write('\t</owner>\n')
    xml.write('\t<size>\n')
    xml.write('\t\t<width>' + str(img_size[0]) + '</width>\n')
    xml.write('\t\t<height>' + str(img_size[1]) + '</height>\n')
    xml.write('\t\t<depth>' + str(img_size[2]) + '</depth>\n')
    xml.write('\t</size>\n')
    xml.write('\t\t<segmented>0</segmented>\n')
    if len(label) == 0:
        print("no object")
        # xml.write('\t<object>\n')
        # xml.write('\t\t<name>False</name>\n')
        # xml.write('\t\t<pose>Unspecified</pose>\n')
        # xml.write('\t\t<truncated>0</truncated>\n')
        # xml.write('\t\t<difficult>0</difficult>\n')
        # xml.write('\t\t<bndbox>\n')
        # xml.write('\t\t\t<xmin>' + str(0) + '</xmin>\n')
        # xml.write('\t\t\t<ymin>' + str(0) + '</ymin>\n')
        # xml.write('\t\t\t<xmax>' + str(0) + '</xmax>\n')
        # xml.write('\t\t\t<ymax>' + str(0) + '</ymax>\n')
        # xml.write('\t\t</bndbox>\n')
        # xml.write('\t</object>\n')
    else:
        for i,indi_label in enumerate(label):
            xml.write('\t<object>\n')
            # if indi_label[2] - indi_label[0] > indi_label[3] - indi_label[1]:
            #     xml.write('\t\t<name>True_w</name>\n')
            # else:
            #     xml.write('\t\t<name>True_h</name>\n')
            xml.write('\t\t<name>True</name>\n')
            xml.write('\t\t<pose>Unspecified</pose>\n')
            xml.write('\t\t<truncated>0</truncated>\n')
            xml.write('\t\t<difficult>0</difficult>\n')
            xml.write('\t\t<bndbox>\n')
            xml.write('\t\t\t<xmin>' + str(indi_label[0]) + '</xmin>\n')
            xml.write('\t\t\t<ymin>' + str(indi_label[1]) + '</ymin>\n')
            xml.write('\t\t\t<xmax>' + str(indi_label[2]) + '</xmax>\n')
            xml.write('\t\t\t<ymax>' + str(indi_label[3]) + '</ymax>\n')
            xml.write('\t\t</bndbox>\n')
            xml.write('\t</object>\n')
    xml.write('</annotation>')


