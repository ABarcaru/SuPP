# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 17:32:31 2018

@author: Andrei ROG
"""
from scipy.stats import ranksums,mannwhitneyu
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import Text
def eval_signif(ListACC,type_test = 'ranksum',lda=False):
    P1 = []
    P2 = []
    P3 = []
    P4 = []
    P5 = []
    P6 = []
    TotalT = []
    for i,lstacc in enumerate(ListACC):
        spp_var_svm = lstacc[0]
        pca_var_svm = lstacc[1]        
        spp_var_nvb = lstacc[2]
        pca_var_nvb = lstacc[3]
        pls_var_clf = lstacc[4]
        lda_var_clf = lstacc[5]
        T = np.zeros([6,3])
        [st1,p_v_svm_11] = mannwhitneyu(spp_var_svm,pca_var_svm,alternative='less')#between SUPP_SVM and PCA_SVM
        [st1,p_v_svm_12] = mannwhitneyu(spp_var_svm,pca_var_svm,alternative='two-sided')
        [st1,p_v_svm_13] = mannwhitneyu(spp_var_svm,pca_var_svm,alternative='greater')
        T[0,:] = np.array([p_v_svm_11,p_v_svm_12,p_v_svm_13])
        [st2,p_v_svm_21] = mannwhitneyu(spp_var_svm,pls_var_clf,alternative='less')#between SUPP_SVM and PLS-DA
        [st2,p_v_svm_22] = mannwhitneyu(spp_var_svm,pls_var_clf,alternative='two-sided')
        [st2,p_v_svm_23] = mannwhitneyu(spp_var_svm,pls_var_clf,alternative='greater')
        T[1,:] = np.array([p_v_svm_21,p_v_svm_22,p_v_svm_23])
        [st3,p_v_svm_31] = mannwhitneyu(spp_var_svm,lda_var_clf,alternative='less')#between SUPP_SVM and LDA
        [st3,p_v_svm_32] = mannwhitneyu(spp_var_svm,lda_var_clf,alternative='two-sided')
        [st3,p_v_svm_33] = mannwhitneyu(spp_var_svm,lda_var_clf,alternative='greater')
        T[2,:] = np.array([p_v_svm_31,p_v_svm_32,p_v_svm_33])
        [st3,p_v_nvb_11] = mannwhitneyu(spp_var_nvb,pca_var_nvb,alternative='less')#between SUPP_NB and PCA_NB
        [st3,p_v_nvb_12] = mannwhitneyu(spp_var_nvb,pca_var_nvb,alternative='two-sided')
        [st3,p_v_nvb_13] = mannwhitneyu(spp_var_nvb,pca_var_nvb,alternative='greater')
        T[3,:] = np.array([p_v_nvb_11,p_v_nvb_12,p_v_nvb_13])
        [st4,p_v_nvb_21] = mannwhitneyu(spp_var_nvb,pls_var_clf,alternative='less')#between SUPP_NB and PLS-DA
        [st4,p_v_nvb_22] = mannwhitneyu(spp_var_nvb,pls_var_clf,alternative='two-sided')
        [st4,p_v_nvb_23] = mannwhitneyu(spp_var_nvb,pls_var_clf,alternative='greater')
        T[4,:] = np.array([p_v_nvb_21,p_v_nvb_22,p_v_nvb_23])
        [st5,p_v_nvb_31] = mannwhitneyu(spp_var_nvb,lda_var_clf,alternative='less')#between SUPP_NB and LDA
        [st5,p_v_nvb_32] = mannwhitneyu(spp_var_nvb,lda_var_clf,alternative='two-sided')
        [st5,p_v_nvb_33] = mannwhitneyu(spp_var_nvb,lda_var_clf,alternative='greater')
        T[5,:] = np.array([p_v_nvb_31,p_v_nvb_32,p_v_nvb_33])
        TotalT.append(T)
        P1.append(p_v_svm_12)
        P2.append(p_v_svm_22)
        P3.append(p_v_svm_32)
        P4.append(p_v_nvb_12)
        P5.append(p_v_nvb_22)
        P6.append(p_v_nvb_32)
        
#        if lda == True:
#            lda_var = lstacc[6]
#            [st,p_v_svm_3] = mannwhitneyu(spp_var_svm,lda_var,alternative='greater')
#            [st,p_v_nvb_3] = mannwhitneyu(spp_var_nvb,lda_var,alternative='greater')
#            P5.append(p_v_svm_3)
#            P6.append(p_v_nvb_3)
#    if lda == True:
    return P1,P2,P3,P4,P5,P6,TotalT
#    else:
#        return P1,P2,P3,P4
if __name__ == '__main__':
    pValColor = [0.5,0.5,0.5]
    labelSize = 16
    titleSize = 18
    xTickSize = 14
    Fractions = np.linspace(0.16,0.72,5)
    # ---- for 3 component ---###
    P1,P2,P3,P4,P5,P6,Tt1 = eval_signif(L1,type_test = 'ranksum')#Wine
    P7,P8,P9,P10,P11,P12,Tt2 = eval_signif(L2,type_test = 'ranksum')#IRIS
    Pc1,Pc2,Pc3,Pc4,Pc5,Pc6,Tt3 = eval_signif(L3,type_test = 'ranksum')#carcinoma
    f, ((ax1, ax2), (ax3, ax4),(ax5, ax6)) = plt.subplots(3,2,sharey=False)
    ###### WINE DATA#########
    # for the SVM
    ax1.plot(Fractions,-np.log10(0.05)*np.ones(len(Fractions)),linestyle = ':',color = pValColor,label='p-value = 0.05')
    ax1.plot(Fractions,-np.log10(P1),'-o',label='SuPP-SVM vs PCA-SVM')
    ax1.plot(Fractions,-np.log10(P2),'-o',label='SuPP-SVM vs PLS-DA')
    ax1.plot(Fractions,-np.log10(P3),'-o',label='SuPP-SVM vs LDA')     
    ax1.set_xlabel('# training point / # validation points')
    ax1.set_ylabel('-log10(p-value)')
    ax1.legend()#loc='right'
    ax1.set_title('Wine data set, components = 3, SVM')
    ax1.title.set_fontsize(titleSize)
    ax1.xaxis.label.set_fontsize(labelSize)
    # for the bayesian
    ax2.plot(Fractions,-np.log10(0.05)*np.ones(len(Fractions)),linestyle = ':',color = pValColor,label='p-value = 0.05')
    ax2.plot(Fractions,-np.log10(P4),'-o',label='SuPP-NB vs PCA-NB')
    ax2.plot(Fractions,-np.log10(P5),'-o',label='SuPP-NB vs PLS-DA')
    ax2.plot(Fractions,-np.log10(P6),'-o',label='SuPP-NB vs PLS-DA')    
    ax2.set_xlabel('# training point / # validation points')
    ax2.set_ylabel('-log10(p-value)')
    ax2.legend()#loc='right'
    ax2.set_title('Wine data set, components = 3, Bayes')
    ax2.title.set_fontsize(titleSize)
    ax2.xaxis.label.set_fontsize(labelSize)
    ###### IRIS DATA#########
    # for the SVM
    ax3.plot(Fractions,-np.log10(0.05)*np.ones(len(Fractions)),linestyle = ':',color = pValColor,label='p-value = 0.05')
    ax3.plot(Fractions,-np.log10(P7),'-o',label='SuPP-SVM vs PCA-SVM')
    ax3.plot(Fractions,-np.log10(P8),'-o',label='SuPP-SVM vs PLS-DA')
    ax3.plot(Fractions,-np.log10(P9),'-o',label='SuPP-SVM vs LDA')     
    ax3.set_xlabel('# training point / # validation points')
    ax3.set_ylabel('-log10(p-value)')
    ax3.legend()#loc='right'
    ax3.set_title('Iris data set, components = 3, SVM')
    ax3.title.set_fontsize(titleSize)
    ax3.xaxis.label.set_fontsize(labelSize)
    # for the bayesian
    ax4.plot(Fractions,-np.log10(0.05)*np.ones(len(Fractions)),linestyle = ':',color = pValColor,label='p-value = 0.05')
    ax4.plot(Fractions,-np.log10(P10),'-o',label='SuPP-NB vs PCA-NB')
    ax4.plot(Fractions,-np.log10(P11),'-o',label='SuPP vs PLS-DA')
    ax4.plot(Fractions,-np.log10(P12),'-o',label='SuPP vs LDA')     
    ax4.set_xlabel('# training point / # validation points')
    ax4.set_ylabel('-log10(p-value)')
    ax4.legend()#loc='right'
    ax4.set_title('Iris data set, components = 3, Bayes')
    ax4.title.set_fontsize(titleSize)
    ax4.xaxis.label.set_fontsize(labelSize)
    ###### CARCINOMA DATA#########
    # for the SVM
    ax5.plot(Fractions,-np.log10(0.05)*np.ones(len(Fractions)),linestyle = ':',color = pValColor,label='p-value = 0.05')
    ax5.plot(Fractions,-np.log10(Pc1),'-o',label='SuPP-SVM vs PCA-SVM')
    ax5.plot(Fractions,-np.log10(Pc2),'-o',label='SuPP-SVM vs PLS-DA')
    ax5.plot(Fractions,-np.log10(Pc3),'-o',label='SuPP-SVM vs LDA')     
    ax5.set_xlabel('# training point / # validation points')
    ax5.set_ylabel('-log10(p-value)')
    ax5.legend()#loc='right'
    ax5.set_title('Carcinoma data set, components = 3, SVM')
    ax5.title.set_fontsize(titleSize)
    ax5.xaxis.label.set_fontsize(labelSize)
    # for the bayesian
    ax6.plot(Fractions,-np.log10(0.05)*np.ones(len(Fractions)),linestyle = ':',color = pValColor,label='p-value = 0.05')
    ax6.plot(Fractions,-np.log10(Pc4),'-o',label='SuPP-NB vs PCA-NB')
    ax6.plot(Fractions,-np.log10(Pc5),'-o',label='SuPP vs PLS-DA')
    ax6.plot(Fractions,-np.log10(Pc6),'-o',label='SuPP vs LDA')     
    ax6.set_xlabel('# training point / # validation points')
    ax6.set_ylabel('-log10(p-value)')
    ax6.legend()#loc='right'
    ax6.set_title('Carcinoma data set, components = 3, Bayes')
    ax6.title.set_fontsize(titleSize)
    ax6.xaxis.label.set_fontsize(labelSize)
    ##################################################
    # ---- for 2 component ---                     ###
    ##################################################
    P12,P22,P32,P42,P52,P62,Tt12 = eval_signif(L12,type_test = 'ranksum',lda=True)
    P72,P82,P92,P102,P112,P122,Tt22 = eval_signif(L22,type_test = 'ranksum',lda=True)
    Pc12,Pc22,Pc32,Pc42,Pc52,Pc62,Tt32 = eval_signif(L32,type_test = 'ranksum',lda=True)
    f2, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2,sharey=False)
    ###### WINE DATA#########
    # for the SVM
    ax1.plot(Fractions,-np.log10(0.05)*np.ones(len(Fractions)),linestyle = ':',color = pValColor,label='p-value = 0.05')
    ax1.plot(Fractions,-np.log10(P12),'-o',label='SuPP-SVM vs PCA-SVM')
    ax1.plot(Fractions,-np.log10(P22),'-o',label='SuPP-SVM vs PLS-DA')
    ax1.plot(Fractions,-np.log10(P32),'-o',label='SuPP vs LDA')    
    ax1.set_xlabel('# training point / # validation points')
    ax1.set_ylabel('-log10(p-value)')
    ax1.legend()#loc='right'
    ax1.set_title('Wine data set, components = 2, SVM')
    ax1.title.set_fontsize(titleSize)
    ax1.xaxis.label.set_fontsize(labelSize)
    # for the bayesian
    ax2.plot(Fractions,-np.log10(0.05)*np.ones(len(Fractions)),linestyle = ':',color = pValColor,label='p-value = 0.05')
    ax2.plot(Fractions,-np.log10(P42),'-o',label='SuPP-NB vs PCA-NB')
    ax2.plot(Fractions,-np.log10(P52),'-o',label='SuPP-NB vs PLS-DA')
    ax2.plot(Fractions,-np.log10(P62),'-o',label='SuPP-NB vs LDA')    
    ax2.set_xlabel('# training point / # validation points')
    ax2.set_ylabel('-log10(p-value)')
    ax2.legend()#loc='right'
    ax2.set_title('Wine data set, components = 2, Bayes')
    ax2.title.set_fontsize(titleSize)
    ax2.xaxis.label.set_fontsize(labelSize)
    ###### IRIS DATA#########
    # for the SVM
    ax3.plot(Fractions,-np.log10(0.05)*np.ones(len(Fractions)),linestyle = ':',color = pValColor,label='p-value = 0.05')
    ax3.plot(Fractions,-np.log10(P72),'-o',label='SuPP-SVM vs PCA-SVM')
    ax3.plot(Fractions,-np.log10(P82),'-o',label='SuPP-SVM vs PLS-DA')
    ax3.plot(Fractions,-np.log10(P92),'-o',label='SuPP-SVM vs LDA')    
    ax3.set_xlabel('# training point / # validation points')
    ax3.set_ylabel('-log10(p-value)')
    ax3.legend()#loc='right'
    ax3.set_title('Iris data set, components = 2, SVM')
    ax3.title.set_fontsize(titleSize)
    ax3.xaxis.label.set_fontsize(labelSize)
    # for the bayesian
    ax4.plot(Fractions,-np.log10(0.05)*np.ones(len(Fractions)),linestyle = ':',color = pValColor,label='p-value = 0.05')
    ax4.plot(Fractions,-np.log10(P102),'-o',label='SuPP-NB vs PCA')
    ax4.plot(Fractions,-np.log10(P112),'-o',label='SuPP-NB vs PLS-DA')
    ax4.plot(Fractions,-np.log10(P122),'-o',label='SuPP-NB vs LDA')    
    ax4.set_xlabel('# training point / # validation points')
    ax4.set_ylabel('-log10(p-value)')
    ax4.legend()#loc='right'
    ax4.set_title('Iris data set, components = 2, Bayes')
    ax4.title.set_fontsize(titleSize)
    ax4.xaxis.label.set_fontsize(labelSize)
    ###### CARCINOMA DATA#########
    # for the SVM
    ax5.plot(Fractions,-np.log10(0.05)*np.ones(len(Fractions)),linestyle = ':',color = pValColor,label='p-value = 0.05')
    ax5.plot(Fractions,-np.log10(Pc12),'-o',label='SuPP-SVM vs PCA-SVM')
    ax5.plot(Fractions,-np.log10(Pc22),'-o',label='SuPP-SVM vs PLS-DA')
    ax5.plot(Fractions,-np.log10(Pc32),'-o',label='SuPP-SVM vs LDA')    
    ax5.set_xlabel('# training point / # validation points')
    ax5.set_ylabel('-log10(p-value)')
    ax5.legend()#loc='right'
    ax5.set_title('Carcinoma data set, components = 2, SVM')
    ax5.title.set_fontsize(titleSize)
    ax5.xaxis.label.set_fontsize(labelSize)
    # for the bayesian
    ax6.plot(Fractions,-np.log10(0.05)*np.ones(len(Fractions)),linestyle = ':',color = pValColor,label='p-value = 0.05')
    ax6.plot(Fractions,-np.log10(Pc42),'-o',label='SuPP-NB vs PCA')
    ax6.plot(Fractions,-np.log10(Pc52),'-o',label='SuPP-NB vs PLS-DA')
    ax6.plot(Fractions,-np.log10(Pc62),'-o',label='SuPP-NB vs LDA')    
    ax6.set_xlabel('# training point / # validation points')
    ax6.set_ylabel('-log10(p-value)')
    ax6.legend()#loc='right'
    ax6.set_title('Carcinoma data set, components = 2, Bayes')
    ax6.title.set_fontsize(titleSize)
    ax6.xaxis.label.set_fontsize(labelSize)
    ##########################################
    #             LDA                        #
    ##########################################
    
#    f3, (ax1, ax2) = plt.subplots(1,2,sharey=False)
#    ax1.plot(Fractions,-np.log10(0.05)*np.ones(len(Fractions)),linestyle = ':',color = pValColor,label='p-value = 0.05')
#    ax1.plot(Fractions,-np.log10(PLDA1),'-o',label='SuPP-SVM vs LDA')
#    ax1.plot(Fractions,-np.log10(PLDA2),'-o',label='SuPP-Bayes vs LDA')    
#    ax1.set_xlabel('# training point / # validation points')
#    ax1.set_ylabel('-log10(p-value)')
#    ax1.legend()#loc='right'
#    ax1.set_title('Wine data set, components = 2')
#    ax1.title.set_fontsize(titleSize)
#    ax1.xaxis.label.set_fontsize(labelSize)
#    # for the bayesian
#    ax2.plot(Fractions,-np.log10(0.05)*np.ones(len(Fractions)),linestyle = ':',color = pValColor,label='p-value = 0.05')
#    ax2.plot(Fractions,-np.log10(PLDA12),'-o',label='SuPP-SVM vs LDA')
#    ax2.plot(Fractions,-np.log10(PLDA22),'-o',label='SuPP-Bayes vs LDA')    
#    ax2.set_xlabel('# training point / # validation points')
#    ax2.set_ylabel('-log10(p-value)')
#    ax2.legend()#loc='right'
#    ax2.set_title('Iris data set, components = 2')
#    ax2.title.set_fontsize(titleSize)
#    ax2.xaxis.label.set_fontsize(labelSize)
    
    
    