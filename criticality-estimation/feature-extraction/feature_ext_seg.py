import skimage
from skimage import data, util
import nibabel as nib
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
import scipy.misc
import pandas as pd

from skimage.feature import hog
from skimage import data, exposure
import numpy as np
from skimage.measure import label

import os
import csv

def fractal_dimension(Z, threshold=0.9):
    #print(Z.shape)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        return len(np.where((S > 0) & (S < k*k))[0])


    Z = (Z < threshold)


    p = min(Z.shape)

    n = 2**np.floor(np.log(p)/np.log(2))

    n = int(np.log(n)/np.log(2))
    #print(n)
    sizes = 2**np.arange(n, 1, -1)

    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))
    #print(counts)
    return counts
    #coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    #return -coeffs[0]

root_dir_train = "/media/mmlab/data/sesh/Data_ICH/Sesh_Segmentation"
#df = pd.DataFrame(columns=['Centroid1','Centroid2','Centroid3','MajorAxisLength','MinorAxisLength','FirstAxis1','FirstAxis2','FirstAxis3','SecondAxis1','SecondAxis2','SecondAxis3','ThirdAxis1','ThirdAxis2','ThirdAxis3','Eigen1','Eigen2','Eigen3','kurtosis','histogram','bb1','bb2','bb3','bb4','bb5','bb6','entropy','extent','diameter','solidity','f1','f2','f3','f4','f5','f6',])
df = pd.DataFrame(columns=['File,''Age','GCS','Onset','CTA','Centroid1','Centroid2','Centroid3','MajorAxisLength','MinorAxisLength','FirstAxis1','FirstAxis2','FirstAxis3','SecondAxis1','SecondAxis2','SecondAxis3','ThirdAxis1','ThirdAxis2','ThirdAxis3','Eigen1','Eigen2','Eigen3','kurtosis','histogram','bb1','bb2','bb3','bb4','bb5','bb6','entropy','extent','diameter','solidity','f1','f2','f3','f4','f5','f6',])
#dir_t1c = '/home/navodini/Documents/NUS/Radiogenomic/Navodini-Data-Seg/'

excel_file = '/media/mmlab/data/sesh/GAN_Proj/UNet/Features/ICH_Clinical_info.xlsx'
total=90
features = pd.read_excel(excel_file)
count=0
Age = 0
GCA = 0
Onset = 0
CTA = 0
#Sex = ''

mri_name = ''            
for mri_file in sorted(os.listdir(root_dir_train)):
    mri_seg = root_dir_train+'/'+mri_file 
    #print(mri_seg)
    for sub_file in os.listdir(mri_seg):
        #print (sub_file)
        fin_dir = mri_seg+'/'+sub_file
        mri_name = fin_dir[-27:-17]
        #print("FIN DIR",fin_dir[-27:-17], type(fin_dir), type(mri_name))
        mri_name = fin_dir[-27:-17]
        #print("MRI NAME OUT:", mri_name)
        if(fin_dir[-20:-17]=='seg' and fin_dir[-22:-21]=='1'):
            print(fin_dir)
            i=1
            while i<total:
                if(int(features['old_ID'][i])==int(fin_dir[-27:-24])):
                    Age = features['Age'][i]
                    GCS = features['PredictA_GCS'][i]
                    Onset = features['PredictA_Onset to CT'][i]
                    CTA = features['PredictA_CTA Spot'][i]
                    #Sex = features['Sex'][i]
                    #print(features['old_ID'][i], fin_dir[-27:-24], Age, GCS, Onset, CTA)
                    i=total+1
                #else:
                #    print("Not Found", features['old_ID'][i], fin_dir[-27:-24])
                i+=1
            _img = nib.load(fin_dir).get_data()
            #print(np.unique(_img))
            _img = _img.transpose(2,0,1)
            #print(_img.shape)
            #img = np.zeros((155,240,240))

            #img = _img[a0:a0+138, a1:a1+186, a1:a1+186]
            #label_img = label(_img, connectivity=_img.ndim)
            _img[_img>0]=1
            _img[_img<=0]=0
            label_img = label(_img)
            #print("Label",label_img.shape)
            #print(np.unique(_img), np.unique(label_img),np.squeeze(label_img).shape)
            props = skimage.measure.regionprops(label_image = label_img)
            Extent = props[0]['extent']
            Diameter = props[0]['equivalent_diameter']
            Solidity = props[0]['solidity']
            Centroid = props[0]['Centroid']
            BBox = props[0]['bbox']
            #print("BBox",BBox)
            MajorAxisLength = props[0]['major_axis_length']
            MinorAxisLength = props[0]['minor_axis_length']
            #MeanIntensity = props[0]['mean_intensity']
            Vectors = props[0]['inertia_tensor']
            x_inertia = Vectors[0]
            y_inertia = Vectors[1]
            z_inertia = Vectors[2]
            EigenValues = props[0]['inertia_tensor_eigvals']
            kurt = (kurtosis(kurtosis(kurtosis(_img, fisher=True))))    
            #print(kurt)
            histo=0
            i=0
            for i in range(label_img.shape[0]):
                fd, hog_image = hog(_img[i,:,:], orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=True, multichannel=False)
                #print(np.sum(hog_image))
                histo = histo+np.sum((hog_image))
            histo = histo/1000
            #print(histo)
            FractalDim = fractal_dimension(_img)
            Entropy = skimage.measure.shannon_entropy(label_img, base=2)
            #print(Entropy)
            parameters = []
            #print("MRI_NAME:")
            #parameters.append(mri_name)
            #print("Centroid-3", Centroid)
            #parameters.append(fin_dir[-27:-21])
            parameters.append(Age)
            parameters.append(GCS)
            #parameters.append(Sex)
            parameters.append(Onset)
            parameters.append(CTA)
            parameters.extend(list(Centroid))
            #print("MajAxiLen-1", MajorAxisLength)
            parameters.append(MajorAxisLength)
            #print("MinorAxiLen-1", MinorAxisLength)
            parameters.append(MinorAxisLength)
            #print("First Axis", x_inertia)
            parameters.extend(list(x_inertia))
            #print("Second Axis", y_inertia)
            parameters.extend(list(y_inertia))
            #print("Third Axis", z_inertia)
            parameters.extend(list(z_inertia))
            #print("Eigen", EigenValues)
            parameters.extend(list(EigenValues))
            #print("Kurt", kurt)
            parameters.append(kurt)
            #print("Hist", histo)
            parameters.append(histo)
            #print("BBox", BBox)
            parameters.extend(list(BBox))
            #print("Entropy", Entropy)
            parameters.append(Entropy)
            #print("extent", Extent) 
            parameters.append(Extent)
            #print("Diameter", Diameter)
            parameters.append(Diameter)
            #print("Solidity", Solidity)
            parameters.append(Solidity)
            #parameters.append(MeanIntensity)
            #print("Fract", FractalDim)
            parameters.extend(FractalDim)
            #print(parameters)
            parameters=np.asarray(parameters)
            #print((parameters))
            np.save(mri_name,parameters)
            #parameters = pd.DataFrame([parameters],columns=df.columns)     
            #df.to_csv('features_extracted_'+fin_dir[-27:-17]+'.csv', index = None, header=True)
            #df = df.append(parameters) 
            '''with open('ICHFeatures.csv', 'a') as fp:
                print("Writing to excel")
                wr = csv.writer(fp, dialect='excel')
                wr.writerow(parameters)
            '''
