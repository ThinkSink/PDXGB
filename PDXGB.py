import numpy as np

from matplotlib import pyplot
from numpy import interp
from xgboost import XGBClassifier

from keras.models import Model
from keras.layers import Dense, Input, Activation,Multiply


from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import time

from sklearn.ensemble import RandomForestClassifier
from numpy import matlib as mb
from sklearn.decomposition import PCA
from sklearn import preprocessing
from keras.utils import to_categorical
def DeepAE(x_train):
    encoding_dim = 128
    input_img = Input(shape=(1372,))

    # encoder layers
    encoded = Dense(512, activation='relu')(input_img)
    encoded = Dense(256, activation='relu')(encoded)
    encoder_output = Dense(encoding_dim)(encoded)

    # decoder layers
    decoded = Dense(256, activation='relu')(encoder_output)
    decoded = Dense(512, activation='relu')(decoded)
    decoded = Dense(1372, activation='tanh')(decoded)

    # construct the autoencoder model
    autoencoder = Model(inputs=input_img, outputs=decoded)
    encoder = Model(inputs=input_img, outputs=encoder_output)

    # compile autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')

    # training

    autoencoder.fit(x_train, x_train, epochs=20, batch_size=128, shuffle=True)

    # plotting
    encoded_imgs = encoder.predict(x_train)




    return encoder_output, encoded_imgs



def transfer_label_from_prob(proba):
    label = [1 if val >= 0.5 else 0 for val in proba]
    return label

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = preprocessing.LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y =to_categorical(y)
    return y, encoder

def calculate_performace(test_num, pred_y, labels): # pred_y = proba, labels = real_labels
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    acc = float(tp + tn) / test_num

    if tp == 0 and fp == 0:
        precision = 0
        MCC = 0
        f1_score = 0
        sensitivity = float(tp) / (tp + fn)
        specificity = float(tn) / (tn + fp)
    else:
        precision = float(tp) / (tp + fp)
        sensitivity = float(tp) / (tp + fn)
        specificity = float(tn) / (tn + fp)
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        f1_score = float(2 * tp) / ((2 * tp) + fp + fn)
    return acc, precision, sensitivity, specificity, MCC, f1_score

def PDXGB():

    MM = np.loadtxt(r".\dataset\dataset1\miRNA similarity matrix.txt", delimiter="\t")
    SM = np.loadtxt(r".\dataset\dataset1\SM similarity matrix.txt", delimiter="\t")
    A = np.loadtxt(r".\dataset\dataset1\known miRNA-SM association matrix.txt", dtype=int, delimiter="\t")

    mm = np.repeat(MM, repeats=831, axis=0)

    sm = mb.repmat(SM, 541, 1)
    H = np.concatenate((sm, mm), axis=1)  # (449571,1372)

    label = A.reshape((449571, 1))

    # H, label = prepare_data2()
    y, encoder = preprocess_labels(label)
    num = np.arange(len(y))
    y = y[num]



    encoder, H_data = DeepAE(H)



    pca = PCA(n_components=128, random_state=1)
    X2 = pca.fit_transform(H)
    H_data = np.hstack((H_data, X2))
    H_data = np.array(H_data)





    t = 0
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    num_cross = 5
    # num_cross = 10
    aucs = []
    all_performance = []



    for fold in range(num_cross):
        train = np.array([x for i, x in enumerate(H_data) if i % num_cross != fold])
        test = np.array([x for i, x in enumerate(H_data) if i % num_cross == fold])
        train_label = np.array([x for i, x in enumerate(y) if i % num_cross != fold])
        test_label = np.array([x for i, x in enumerate(y) if i % num_cross == fold])

        real_labels = []
        for val in test_label:
            if val[0] == 1:  # tuples in array, val[0]- first element of tuple
                real_labels.append(0)
            else:
                real_labels.append(1)

        train_label_new = []
        for val in train_label:
            if val[0] == 1:
                train_label_new.append(0)
            else:
                train_label_new.append(1)

        prefilter_train = train
        prefilter_test = test



        clf = XGBClassifier(n_estimators=100, learning_rate=0.3)

        clf.fit(prefilter_train, train_label_new)  # Training
        ae_y_pred_prob = clf.predict_proba(prefilter_test)[:, 1]  # testing
        proba = transfer_label_from_prob(ae_y_pred_prob)

        acc, precision, sensitivity, specificity, MCC, f1_score = calculate_performace(len(real_labels), proba,
                                                                                       real_labels)


        fpr, tpr, auc_thresholds = roc_curve(real_labels, ae_y_pred_prob)
        auc_score = auc(fpr, tpr)
        aucs.append(auc_score)

        precision1, recall, pr_threshods = precision_recall_curve(real_labels, ae_y_pred_prob)
        aupr_score = auc(recall, precision1)
        print("PDXGB:", acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score, f1_score)
        all_performance.append([acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score, f1_score])



        t = t + 1  # AUC fold number

        pyplot.plot(fpr, tpr, label='ROC fold %d (AUC = %0.4f)' % (t, auc_score))


        mean_tpr += interp(mean_fpr, fpr, tpr)  # one dimensional interpolation
        mean_tpr[0] = 0.0

        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.title('ROC curve: 5-Fold CV')
        pyplot.legend()





#
    mean_tpr /= num_cross


    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    np.savetxt('rf_fpr.csv', mean_fpr, delimiter=',')
    np.savetxt('rf_tpr.csv', mean_tpr, delimiter=',')



    pyplot.plot(mean_fpr, mean_tpr, linewidth=1, alpha=0.8,
                label='Mean ROC(AUC = %0.4f)' % mean_auc)

    pyplot.legend()

    print('std_auc=', std_auc)
    plt.savefig('5-fold PDXGB(AUC = %0.4f).png' % mean_auc, dpi=300)

    pyplot.show()
    print('*******AUTO-STB*****')
    print('mean performance of XGB using raw feature')
    print(np.mean(np.array(all_performance), axis=0))
    Mean_Result = np.mean(np.array(all_performance), axis=0)
    print('---' * 20)
    print('Mean-Accuracy=', Mean_Result[0], '\n Mean-precision=', Mean_Result[1])
    print('Mean-Sensitivity=', Mean_Result[2], '\n Mean-Specificity=', Mean_Result[3])
    print('Mean-MCC=', Mean_Result[4], '\n' 'Mean-auc_score=', Mean_Result[5])
    print('Mean-Aupr-score=', Mean_Result[6], '\n' 'Mean_F1=', Mean_Result[7])
    print('---' * 20)


if __name__=="__main__":
    time_start = time.time()

    PDXGB()

    time_end = time.time()
    print('time cost', time_end - time_start, 's')
