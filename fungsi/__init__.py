import cv2
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

props = [
        "energy",
        "homogeneity",
        "contrast",
        "correlation"
        ]
degrees = [0,45,90,135]


def glcm(img):
    row = []
    value = []
    upload = img.split("/")
    file = upload[-1]
    img_read = cv2.imread(img,0)
    name_gray = "./tmp/img/"+file[:-4]+" gray.jpg"
    cv2.imwrite(name_gray,img_read)
    image_disp = {
        'original'  : img,
        'gray'      : name_gray,
    }
    for degree in degrees:
        glcm = graycomatrix(img_read, [5], [degree], symmetric=True, normed=True)
        ro = []
        ro.append(degree)
        for prop in props:
            ro.append(float(graycoprops(glcm, prop)))
            row.append(float(graycoprops(glcm, prop)))
        value.append(ro)
    
    return image_disp, value, pd.DataFrame([row],columns=kolom())

def dataframe_glcm(l_data):
    data = pd.DataFrame()
    for jenis in l_data.keys():
        for isi in l_data[jenis]:
            l, x,res = glcm(isi)
            res['label'] = jenis
            data = pd.concat([data,res],ignore_index=True)
    return data

def get_accuraacy(l_data):
    test = dataframe_glcm(l_data)
    testX = test.drop('label',axis=1)
    testy = test['label']
    hasilX = clf.predict(testX)
    confus = confusion_matrix(testy,hasilX)
    ConfusionMatrixDisplay(confus,display_labels=('benign', 'malignant')).plot()
    name_confus = "tmp/img/confus.jpg"
    plt.savefig(name_confus)
    return hasilX, name_confus, accuracy_score(testy,hasilX)


def kolom():
    col = []
    for degree in degrees:
        for prop in props:
            col.append(prop+'('+str(degree)+')')
    return col

glcm_df = pd.read_csv("tmp/glcm_no_index.csv")
X = glcm_df.drop('label',axis=1)
Y = glcm_df['label']
clf = svm.SVC(decision_function_shape='ovr')
clf.fit(X, Y)

# train_glcm()
def predict_glcm(img):
    l, x,res = glcm(img)
    return l,x, clf.predict(res)[0]

