
import numpy as np
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('re.jpg')
#  画出统计直方图
#img = cv2.Canny(img,30,110)
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
#plt.subplot(111)
plt.plot(histr,color = 'g')
plt.title('gray_hist')
plt.xlim([0,256])
#pos = plt.ginput(1)
#print(pos)
img_erode = np.array(cv2.imread('erosion.jpg'))
img_erode0 = np.zeros((img_erode.shape[0],img_erode.shape[1]),dtype = int)
img_erode0 = img_erode[:,:,1]
for i in range(img_erode.shape[0]):
    for j in range(img_erode.shape[1]):
        if img_erode0[i][j] < 127:
            img_erode0[i][j] = 0
        else:
            img_erode0[i][j] = 1
#print(img_erode.shape)
row,col,m = img_erode.shape
#plt.figure(3)
#plt.imshow(img_erode)
[y0,x0] = [2,456]#  初始化种子点
counter = 0
#生长函数
def regionGrow(x,y,img1):
    row,col,m = img.shape
    stack = np.zeros((img1.shape[0]*img1.shape[1],2),dtype = int)
    pstack = 0
    stack[pstack][0] = x
    stack[pstack][1] = y
    img_river = np.zeros((img1.shape[0],img1.shape[1]),dtype = int)
    img_river[x][y] = 255
    counter = 1
    while pstack >= 0:
        x1 = stack[pstack][0]
        y1 = stack[pstack][1]
        pstack = pstack - 1
        for i in range(-1,2):
            for j in range(-1,2):
                #if (x1+i) >= 0 and (x1+i) <= row:
                #and img1[x1+i,y1+j] == img1[x1,y1]and (img_river[x1+i,y1+j] != img_river[x1,y1])
                try:
                    if ((x1+i >= 0) and (x1+i < row) and (y1+j >= 0) and (y1+j < col)) and ((img1[x1+i,y1+j] == img1[x1,y1]) and (img_river[x1+i,y1+j] != img_river[x1,y1])):
                        img_river[x1+i,y1+j] = img_river[x1,y1]
                        counter = counter + 1
                        pstack = pstack + 1
                        stack[pstack][0] = x1 + i
                        stack[pstack][1] = y1 + j
                except:
                    print(x1,y1)
    #print(img_river)
    print(counter)      
    #plt.imshow(img_river,cmap = 'gray')            
    return img_river
img_river = regionGrow(x0,y0,img_erode0)
plt.figure()
plt.imshow(img_river,cmap = 'gray')
plt.show()
#print(counter)  