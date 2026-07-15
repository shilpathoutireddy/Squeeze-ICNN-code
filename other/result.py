import pandas as pd, numpy as np
# from colorama import Fore, init
import matplotlib.pyplot as plt
from other.Confusion_matrix import *
import statistics,cv2
from matplotlib import style
style.use('seaborn-whitegrid')
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
rn=0

def roc_plot_multi(y_test,y_score1,label,y,clr):
    plt.figure()
    for ii in range(len(y_score1)):
        y_score=y_score1[ii][:len(y_test)]
        # Binarize
        y_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
        y_score = label_binarize(y_score, classes=[0, 1, 2, 3, 4])
        n_classes = y_test.shape[1]

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        plt.plot(fpr[2], tpr[2], label=label[ii] + '(area = %0.2f)' % roc_auc[2],color=clr[ii])

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Results/' + str(y) + '/'+str(y)+'_roc.png')
    plt.show(block=False)

def roc_plot(y_test,y_score1,label,y,clr):
    plt.figure()
    for ii in range(len(y_score1)):
        y_score=y_score1[ii][:len(y_test)]

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fpr[0], tpr[0], _ = roc_curve(y_test, y_score)
        roc_auc[0] = auc(fpr[0], tpr[0])
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score)
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        plt.plot(fpr[0], tpr[0], label=label[ii] + '(area = %0.2f)' % roc_auc[0],color=clr[ii])

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Results/' + str(y) + '/'+str(y)+'_roc.png')
    plt.show(block=False)



def stat_analysis(xx):
    mn = np.mean(xx, axis=0).reshape(-1, 1)
    mdn = np.median(xx, axis=0).reshape(-1, 1)
    std_dev = np.std(xx, axis=0).reshape(-1, 1)
    min = np.min(xx, axis=0).reshape(-1, 1)
    mx = np.max(xx, axis=0).reshape(-1, 1)
    return np.concatenate((mn, mdn, std_dev, min, mx), axis=1)

def roc_plot(y_test, y_score1, label, db, cond):
    plt.figure()
    for ii in range(len(y_score1)):
        y_score = y_score1[ii][:len(y_test)]

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fpr[0], tpr[0], _ = roc_curve(y_test, y_score)
        roc_auc[0] = auc(fpr[0], tpr[0])
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score)
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        plt.plot(fpr[0], tpr[0], label=label[ii])

        # plt.plot(fpr[0], tpr[0], label=label[ii] + '(area = %0.2f)' % roc_auc[0])

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Results/DB' + str(db) + '/' + cond + '/' + '_roc.png')
    plt.show(block=False)

def plotresult():
        from colorama import Fore, init
        clmn = ['LSTM', 'DCNN', 'BIGRU', 'SqueezeNet', 'DenseNet','CNN', 'DNN', 'Squeeze-ICNN']
        # clmn = ['CNN', 'BiGRU', 'RNN', 'DBN', 'DeepMaxout', 'DEEPVISION', 'DNN', 'MLP-CNN+LSTM']
        lp1 = ['60', '70', '80', '90']
        db=1

        sspr = np.load('saved data/ssim_psnr' + '.npy')
        df = pd.DataFrame(sspr, columns=['PSNR', 'SSIM'],
                          index=['Meanfilter', 'Gaussianfilter', 'Weinerfilter', 'conventional median filter',
                                 'proposed median filter'])
        print(df)
        df.to_csv('Results/' + 'PSNR_SSIM.csv')

        # dja = np.load('pre_evaluated/saved/seg_ot_dja' + str(1) + '.npy')
        # mtd = ['fcm', 'kmeans', 'Birch','conventional segnet', 'proposed segnet']
        # name2 = ['dice', 'jaccard', 'segmentation accuracy']
        # df2 = pd.DataFrame(dja, columns=mtd, index=name2)
        # print(df2)
        # df2.to_csv('Results/DB' + 'Dice_Jaccard_seg_accuracy.csv')

        # act = np.load('pre_evaluated/saved/Actual-' + cond + lp1[2] + str(db) + '.npy')
        # pred = np.load('pre_evaluated/saved/Predicted-' + cond + lp1[2] + str(db) + '.npy')
        # roc_plot(act, pred, clmn, db, cond)

        val = pd.read_csv('pre_evaluated/Performance-' + str(db) + '.csv', index_col=[0, 1])
        new_ = val.loc[([60, 70, 80, 90], 'accuracy'), :]
        new = new_.values
        aa = stat_analysis(new)

        indx = ['Mean', 'Median', 'Std', 'Min', 'Max']
        d = pd.DataFrame(aa.transpose(), columns=clmn, index=indx)
        d.to_csv('Results/DB' + 'statistical_test.csv')

        # ablation atudy #
        name1 = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F_measure', 'MCC', 'NPV', 'FPR', 'FNR']
        ablation = pd.read_csv(f'pre_evaluated/ablation_study' + str(db) + '.csv', header=0)
        print('\nfeature comparison')
        cla_tab = ablation.iloc[:, 1:].values
        Tab = np.empty([len(name1), cla_tab.shape[1] + 1])
        Tab[:, :Tab.shape[1] - 1] = cla_tab
        Tab[:, Tab.shape[1] - 1] = val.loc[([int(lp1[2])]), :]['PROPOSED'].values
        df = pd.DataFrame(Tab,
                          columns=['conventional preprocess','conventional hierarchy of skeleton','squeezenet+CNN',
                                   'proposed'],
                          index=name1)
        print(df.to_markdown())
        df.to_csv(f'Results/DB' + 'ablation_study' + '.csv')

        # training % varied results #
        indx = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f_measure', 'mcc', 'npv', 'fpr', 'fnr']
        plot_result = pd.read_csv('pre_evaluated/Performance-' + str(db) + '.csv', index_col=[0, 1])
        plot_result.columns = clmn
        for i in range(60, 91, 10):
            avg = plot_result.loc[i, :]
            avg.reset_index(drop=True, level=0)
            # avg.to_csv(f'Results/' + str(i) +'-' +cond+ '.csv')
            if db == 1:
                print('\n\t', Fore.RED + str(i))
            else:
                print('\n\t', Fore.RED + str(i))
            print(avg.to_markdown())

        if db == 1:
            print('\n\t', Fore.MAGENTA + 'Statistical Analysis')
        else:
            print('\n\t', Fore.BLUE + 'Statistical Analysis')
        print(pd.read_csv('Results/DB'+ 'statistical_test.csv', header=0).to_markdown())

        custom_colors = ['#33FF57', '#5733FF', '#33FFFF', 'deeppink', 'tan', '#a39998','y', 'r']
        for idx, jj in enumerate(indx):
            new_ = plot_result.loc[([60, 70, 80, 90], [jj]), :]
            new_.reset_index(drop=True, level=1, inplace=True)
            print('\n')
            print(new_)
            new_.to_csv('Results/DB' + jj + 'Graph.csv')
            ax = new_.plot(figsize=(10, 6), marker='D', linestyle='-', color=custom_colors, linewidth=2, markersize=10)
            plt.xlabel('Training Data(%)', fontweight='bold')
            plt.ylabel(jj.upper(), fontweight='bold')
            # ax = new_.plot(figsize=(10, 6), kind='bar', width=0.8, use_index=True,
            #                rot=0, color=custom_colors,
            #                edgecolor='black')
            # plt.xlabel('Training Data(%)')
            # plt.ylabel(jj.upper())
            # ax.set_facecolor('lightgray')
            plt.grid(True, linestyle='--', color='black', alpha=0.7)
            plt.subplots_adjust(bottom=0.2)
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=5)
            plt.savefig('Results/DB' + jj + '.png')
            plt.show(block=False)








