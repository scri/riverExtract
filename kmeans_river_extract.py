import numpy as np
import cv2
import imp
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#im1 = im.resize((1000,1000))
img = cv2.imread('re.jpg')
#  画出统计直方图
colar = ('b','g','r')
plt.figure(1)
for i ,col in enumerate(colar):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.title("color hist")
plt.figure(2)
img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
histr = cv2.calcHist([img_gray],[0],None,[256],[0,256])
plt.plot(histr,color = 'g')
plt.title('gray_hist')
plt.xlim([0,256])

#kmeans计算
img_data = img/255.0
plt.figure(3)
plt.imshow(img_data,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#kernel=np.ones((4,4,),np.float32)/16
#img_data = cv2.filter2D(img_data, -1, kernel)
#对图片进行一下滤波平滑可能会更好看一点
row,col= img_data.shape[0],img_data.shape[1]
img_data=img_data.reshape((row*col,3))
label = KMeans(n_clusters=4).fit_predict(img_data) # 聚类中心的个数为4

label = label.reshape([row, col]) #*50+50 # 聚类获得每个像素所属的类别

label_loca0 = np.zeros((row,col),dtype = int)
label_loca1 = np.zeros((row,col),dtype = int)
label_loca2 = np.zeros((row,col),dtype = int)
label_loca3 = np.zeros((row,col),dtype = int)
for i in range(row):
    for j in range(col):
        if label[i][j] == 0:
            label_loca0[i][j] = 1
        elif label[i][j] == 1:
            label_loca1[i][j] = 2
        elif label[i][j] == 2:
            label_loca2[i][j] = 3
        else:
            label_loca3[i][j] = 4

            
#label = label.reshape([row, col]) #*50+50 # 聚类获得每个像素所属的类别
plt.figure(4)
plt.subplot(141),plt.imshow(label_loca0*30,cmap = 'gray')
plt.subplot(142),plt.imshow(label_loca1*30,cmap = 'gray')
plt.subplot(143),plt.imshow(label_loca2*30,cmap = 'gray')
plt.subplot(144),plt.imshow(label_loca3*30,cmap = 'gray')
plt.figure(5)
label = label.reshape([row, col]) *50+50 # 聚类获得每个像素所属的类别
plt.imshow(label,cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
