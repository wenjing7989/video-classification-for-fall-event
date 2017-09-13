import numpy as np
import os
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from scipy.misc import imresize

np.random.seed(1234)

# 1-184 falling down, 185-224 squat 225-272 sit 273-320 lie down 321-400 walk
# every 8 video are from the same angle, e.g. 1,9,17
def preprocess_input(x):
    if x.ndim==4:
        x[:, :, :, 0] -= 123.68
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 103.939
    else:
        x[:, :, 0] -= 123.68
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 103.939
    return x/128

def transform(img, r1):
    if r1:
        img = img[:,::-1,:]
    return img

fall = {'train':range(0,96), 'val':range(96,128), 'test':range(128,184)}
#184: 96, 32, 56; 0, [1,0]
squat = {'train':range(184,208), 'val':range(208,216), 'test':range(216,224)}
sit = {'train':range(224,256), 'val':range(256,264), 'test':range(264,272)}
lie = {'train':range(272,304), 'val':range(304,312), 'test':range(312,320)}
walk = {'train':range(320,368), 'val':range(368,376), 'test':range(376,400)}

#216: 112, 40, 64; 1, [0,1]
# train208, val72, test120
class falldata:
    def __init__(self, tvt):
        self.tvt = tvt
        self.image = fall[self.tvt]+squat[self.tvt]+sit[self.tvt]+\
                lie[self.tvt]+walk[self.tvt]
        self.num = len(self.image)

    def generate(self, batch_size, shuffle=True, d2=False, multiL=False):
        #d2 means only one frame is fed
        path = './falldata/'

        if multiL:
            nb_classes = 5
            label = [0]*len(fall[self.tvt])+[1]*len(squat[self.tvt])+\
            [2]*len(sit[self.tvt])+[3]*len(lie[self.tvt])+[4]*len(walk[self.tvt])
        else:
            nb_classes = 2
            label = [0]*len(fall[self.tvt])+[1]*(len(self.image)-len(fall[self.tvt]))

        if shuffle:
            comb = zip(self.image,label)
            np.random.shuffle(comb)
            self.image, label = zip(*comb)

        X, y = [], []
        while True:
            for i in range(len(label)):
                assert (self.image[i]<184)!=label[i]
                temp = []
                imgs=os.listdir(path+str(self.image[i]+1).zfill(3))
                imgs=sorted(imgs)
                if d2:
                    #name = imgs[np.random.choice(5)]
                    name = imgs[4]
                    img = plt.imread(path+str(self.image[i]+1).zfill(3)+'/'+name)
                    temp=imresize(img, [224,224])
                else:
                    r1 = np.random.randint(2)
                    for name in imgs:
                        img = plt.imread(path+str(self.image[i]+1).zfill(3)+'/'+name)
                        if self.tvt=='train':
                            img = transform(img, r1)
                        temp.append(imresize(img, [224,224]))
                # if i%10==0:
                #     plt.imshow(temp)
                #     plt.show()
                #     print(label[i])
                temp = np.float32(temp)
                X.append(preprocess_input(temp))
                y.append(label[i])
                if len(y)==batch_size or i==len(label)-1:
                    tmp_inp = np.array(X)
                    tmp_targets = to_categorical(y,nb_classes)
                    X, y = [], []
                    yield tmp_inp, tmp_targets

    def load_all(self, shuffle=True, d2=False, multiL=False):
        path = './falldata/'

        if multiL:
            nb_classes = 5
            label = [0]*len(fall[self.tvt])+[1]*len(squat[self.tvt])+\
            [2]*len(sit[self.tvt])+[3]*len(lie[self.tvt])+[4]*len(walk[self.tvt])
        else:
            nb_classes = 2
            label = [0]*len(fall[self.tvt])+[1]*(len(self.image)-len(fall[self.tvt]))

        if shuffle:
            comb = zip(self.image,label)
            np.random.shuffle(comb)
            self.image, label = zip(*comb)

        X, y = [], []
        for i in range(len(label)):
            assert (self.image[i]<184)!=label[i]
            temp = []
            imgs=os.listdir(path+str(self.image[i]+1).zfill(3))
            imgs=sorted(imgs)
            if d2:
                #name = imgs[np.random.choice(5)]
                name = imgs[4]
                img = plt.imread(path+str(self.image[i]+1).zfill(3)+'/'+name)
                temp=imresize(img, [224,224])
            else:
                r1 = np.random.randint(2)
                for name in imgs:
                    img = plt.imread(path+str(self.image[i]+1).zfill(3)+'/'+name)
                    if self.tvt=='train':
                        img = transform(img, r1)
                    temp.append(imresize(img, [224,224]))
            temp = np.float32(temp)
            X.append(preprocess_input(temp))
            y.append(label[i])

        tmp_inp = np.array(X)
        tmp_targets = to_categorical(y,nb_classes)
        return tmp_inp, tmp_targets

    def get_steps(self):
        return self.num

if __name__=='__main__':
    data = falldata('val')
#    a=data.generate(25)
