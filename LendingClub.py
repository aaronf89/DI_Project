import pandas as pd
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.ma as ma
import datetime as dt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, PolynomialFeatures
from sklearn import linear_model, datasets
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, grid_search
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.learning_curve import learning_curve
plt.style.use('ggplot')


#Functions to clean up data
def ConvPercent(x):
    return float(x.strip('%').strip(' '))/100.0
    
def ConvertEmpYears(x):
    y=x.replace('+', ' ').replace('<',' ').split(' ')
    
    if len(y[0])==0:
        return 0
    elif y[0]=='n/a':
        return np.nan
    else:
        return int(y[0])
        
def ConvertOwnership(x):
    
    if x=='RENT':
        return 0.0
    elif x=='MORTGAGE':
        return 1.0
    elif x=='OWN':
        return 2.0
    else:
        return np.nan
        
def ConvHist(x):
    
    return 12.*(2015-x.year)+11-x.month

def ConvTerm(x):
    
    return float(x.split(' ')[1])
    
def ConvLoanStatus(x, Trouble=[], Success=[]):

    if x in Trouble:
        return bin(0)
    elif x in Success:
        return bin(1)    
    else:
        return 2
        
def IntQuartiles(x, q1=.25, q2=.5, q3=.75):
    
    if x<q1:
        return 'Tier 1'
    elif x>q1 and x<q2:
        return 'Tier 2'
    elif x>q2 and x<q3:
        return 'Tier 3'
    else:
        return 'Tier 4'
    


#Import data function
def GetData(cols, files):

    frames=[]
    for file in files:
        frames.append(pd.read_csv(file, usecols=cols))
    
    return pd.concat(frames, ignore_index=True).dropna()

#Clean Data
def CleanData(df):
    df['revol_util']=(df['revol_util'].apply(ConvPercent))
    df['emp_length']=(df['emp_length'].apply(ConvertEmpYears))
    df['home_ownership']=(df['home_ownership'].apply(ConvertOwnership))
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])
    df['earliest_cr_line'] = df['earliest_cr_line'].apply(ConvHist)
    df['term'] = df['term'].apply(ConvTerm)
    df['int_rate']=(df['int_rate'].apply(ConvPercent))
    
    #Pick out successful loans and unsuccessful ones
    Trouble=[ 'Charged Off', 'Default', 'Does not meet the credit policy.  Status:Charged Off']
    Success=['Fully Paid', 'Does not meet the credit policy.  Status:Fully Paid']
    
    df['loan_status']=df.loan_status.apply(ConvLoanStatus, Trouble=Trouble, Success=Success)
    
    #Now sort interest rate into quatiles
    Q1,Q2,Q3=(df['int_rate'].describe()).ix[[4, 5,6]]
    df['IntCat']=df.int_rate.apply(IntQuartiles, q1=Q1,q2=Q2,q3=Q3)
    
    #Find the percent lost by any given loan (only relevant to defaulted loans)
    df['Loss']=(df.funded_amnt_inv-df.total_pymnt_inv)
    df['PerLoss']=(df.funded_amnt_inv-df.total_pymnt_inv)/df.funded_amnt_inv
    
    return df

#Plot the distribution of lost principal
def PlotLostPrin(df):

    plt.figure(1)
    plt.clf()
    ax1=plt.hist((df[(df['loan_status']==bin(0)) & (df.funded_amnt_inv>1)].PerLoss).as_matrix()*100, bins=50,  color='crimson', label='Charged off/in default', stacked=True, histtype='barstacked', alpha=1)
    plt.xlabel('Fraction of Initial Loan Amount Lost (in percent)', fontsize=20)
    plt.ylabel('Frequency of Loss', fontsize=20)
    plt.title('Principal Lost When Borrower Fails to Pay', fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()



#Function that plots the portion of defaulted and successful loans in each interest rate quartile
def PlotFractions(df):
    grouped=df.groupby(['loan_status', 'IntCat'])
    PaidRates=(grouped.count()['int_rate'])['0b1']/((grouped.count()['int_rate'])['0b1'].sum())
    LostRates=(grouped.count()['int_rate'])['0b0']/((grouped.count()['int_rate'])['0b0'].sum())

    plt.figure(1)
    plt.clf()
    ax4=LostRates.plot(kind='bar', color='crimson', position=0, width=.25, label='Charged off/in default', fontsize=15)
    ax5=PaidRates.plot(kind='bar', color='burlywood', position=1, width=.25, label= 'Paid in Full', fontsize=15)
    ax4.set_xlabel('Interest Rate Category')
    ax4.set_ylabel('Fraction of Loans')
    ax4.set_title('Fraction of Loans in Each Category')
    plt.legend( loc='best', prop={'size':15})
    ax4.xaxis.label.set_fontsize(20)
    ax4.yaxis.label.set_fontsize(20)
    ax4.title.set_fontsize(20)
    plt.show()
    
    
#This function plots the distribution of defaults versus successful loans for each of the 
#seven features of interest in this dataset
def Visualize2(df):
    
    #Visualize the data
    Lose=df[(df['loan_status']==bin(0)) & (df['IntCat']=='Tier 4')]
    Win=df[(df['loan_status']==bin(1)) & (df['IntCat']=='Tier 4')]
    
    plt.figure(1)
    plt.clf()
    ax2=plt.hist((Win.annual_inc[Win.annual_inc<200000]).as_matrix()/1000, bins=40, normed=1, color='burlywood', label= 'Paid in Full', stacked=True, histtype='barstacked')
    ax1=plt.hist((Lose.annual_inc[Lose.annual_inc<200000]).as_matrix()/1000, bins=40, normed=1, color='crimson', label='Charged off/in default', stacked=True, histtype='barstacked', alpha=.5)
    plt.legend( loc='best', prop={'size':15})
    plt.xlabel('Borrower\'s Income (1000\'s of Dollars)', fontsize=20)
    plt.ylabel('Fraction of Borrowers', fontsize=20)
    plt.title('Borrower Income by Loan Type', fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    
    plt.figure(2)
    plt.clf()
    ax2=plt.hist((Win.revol_util[Win.revol_util<1]).as_matrix()*100., bins=80, normed=1, color='burlywood', label= 'Paid in Full', stacked=True, histtype='barstacked')
    ax1=plt.hist((Lose.revol_util[Lose.revol_util<1]).as_matrix()*100., bins=80, normed=1, color='crimson', label='Charged off/in default', stacked=True, histtype='barstacked', alpha=.5)
    plt.legend( loc='best', prop={'size':15})
    plt.xlabel('Borrower\'s Credit Utilization (in %)', fontsize=20)
    plt.ylabel('Normalized Number of Borrowers', fontsize=20)
    plt.title('Credit Use by Loan Type', fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    
    plt.figure(3)
    plt.clf()
    ax2=plt.hist((Win.earliest_cr_line[Win.earliest_cr_line>0]).as_matrix(), bins=80, normed=1, color='burlywood', label= 'Paid in Full', stacked=True, histtype='barstacked')
    ax1=plt.hist((Lose.earliest_cr_line[Lose.earliest_cr_line>0]).as_matrix(), bins=80, normed=1, color='crimson', label='Charged off/in default', stacked=True, histtype='barstacked', alpha=.5)
    plt.legend( loc='best', prop={'size':15})
    plt.xlabel('Borrower\'s Credit History (in months)', fontsize=20)
    plt.ylabel('Normalized Number of Borrowers', fontsize=20)
    plt.title('Credit History by Loan Type', fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    
    plt.figure(4)
    plt.clf()
    ax2=plt.hist(((Win.dropna()).home_ownership).as_matrix(), bins=3, normed=1, color='burlywood', label= 'Paid in Full', stacked=True, histtype='barstacked')
    ax1=plt.hist(((Lose.dropna()).home_ownership).as_matrix(), bins=3, normed=1, color='crimson', label='Charged off/in default', stacked=True, histtype='barstacked', alpha=.5)
    plt.legend( loc='best', prop={'size':15})
    plt.xlabel('Borrower\'s Living Situation', fontsize=20)
    plt.ylabel('Normalized Number of Borrowers', fontsize=20)
    plt.title('Own, Rent or Mortgage by Loan Type', fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    
    plt.figure(5)
    plt.clf()
    ax2=plt.hist((Win.loan_amnt).as_matrix()/1000., bins=40, normed=1, color='burlywood', label= 'Paid in Full', stacked=True, histtype='barstacked')
    ax1=plt.hist((Lose.loan_amnt).as_matrix()/1000., bins=40, normed=1, color='crimson', label='Charged off/in default', stacked=True, histtype='barstacked', alpha=.5)
    plt.legend( loc='best', prop={'size':15})
    plt.xlabel('Loan Amount (1000\'s of $)', fontsize=20)
    plt.ylabel('Normalized Number of Borrowers', fontsize=20)
    plt.title('Loan Amount by Loan Type', fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    
    
    plt.figure(6)
    plt.clf()
    ax2=plt.hist((Win.int_rate[Win.int_rate>.15]).as_matrix()*100., bins=20, normed=1, color='burlywood', label= 'Paid in Full', stacked=True, histtype='barstacked')
    ax1=plt.hist((Lose.int_rate[Lose.int_rate>.15]).as_matrix()*100., bins=20, normed=1, color='crimson', label='Charged off/in default', stacked=True, histtype='barstacked', alpha=.5)
    plt.legend( loc='best', prop={'size':15})
    plt.xlabel('Loan Interest Rate (in %)', fontsize=20)
    plt.ylabel('Normalized Number of Borrowers', fontsize=20)
    plt.title('Interest Rate by Loan Type', fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    
    plt.figure(7)
    plt.clf()
    ax2=plt.hist(((Win.dropna()).emp_length).as_matrix(), bins=11, normed=1, color='burlywood', label= 'Paid in Full', stacked=True, histtype='barstacked')
    ax1=plt.hist(((Lose.dropna()).emp_length).as_matrix(), bins=11, normed=1, color='crimson', label='Charged off/in default', stacked=True, histtype='barstacked', alpha=.5)
    plt.legend( loc='best', prop={'size':15})
    plt.xlabel('Employment Length (in years)', fontsize=20)
    plt.ylabel('Normalized Number of Borrowers', fontsize=20)
    plt.title('Employment Length by Loan Type', fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    


#Grab relevant features and labels from the data set
def GetFeatures(df, RelFeats, conditions, polyorder=1):

    temp=df[conditions].dropna()

    return scale(PolynomialFeatures(degree=polyorder, include_bias=False).fit_transform(scale(temp[RelFeats]))), temp.loan_status.as_matrix(), temp.index


def GetReport(model, PlotROC, X_test, y_test):

    #Results=pd.DataFrame(zip(RelFeats, np.transpose(model.coef_)))
    accur=model.score(X_test, y_test)
    predicted = model.predict(X_test)
    ConfMat=metrics.confusion_matrix(y_test, predicted)
    Rep=metrics.classification_report(y_test, predicted)
    if (y_test[0]=='0b0') | (y_test[0]=='0b1'):
        Bin_to_Int=np.vectorize(int)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(Bin_to_Int(y_test,2), model.predict_proba(X_test)[:,1])
    else:
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    roc_auc = auc(false_positive_rate, true_positive_rate)
    
    print Rep
    print 'Accuracy = '+str(accur)
    print 'AUC = '+str(roc_auc)

    if PlotROC:
        plt.figure(11)
        plt.clf()
        plt.title('Receiver Operating Characteristic')
        plt.plot(false_positive_rate, true_positive_rate, 'b',
        label='AUC = %0.2f'% roc_auc)
        plt.legend(loc='lower right', fontsize=20)
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([-0.05,1.05])
        plt.ylim([-0.05,1.05])
        plt.ylabel('True Positive Rate', fontsize=20)
        plt.xlabel('False Positive Rate', fontsize=20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()
        
    return accur, ConfMat, Rep, roc_auc

#Function that fits a logistic regression model and summarizes results
def FitLogReg(features, labels, RelFeats,c=10, split=.2, Reports=False, PlotROC=False):
    
    accur=None
    ConfMat=None
    Rep=None
    roc_auc=None
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=split)
    model = LogisticRegression(penalty='l2', C=c, class_weight='auto')
    model = model.fit(X_train, y_train)
    
    if Reports:
        accur, ConfMat, Rep, roc_auc=GetReport(model, PlotROC, X_test, y_test)
        
        
        
    return model, accur, ConfMat, Rep, roc_auc
    
    
#Function that fits a logistic regression model and summarizes results
def FitSVMReg(features, labels, RelFeats, c=10, split=.2, Reports=False, PlotROC=False):
    
    accur=None
    ConfMat=None
    Rep=None
    roc_auc=None
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=split)
    model = SVC(C=c, class_weight='auto', probability=True)
    model = model.fit(X_train, y_train)
    
    if Reports:
        accur, ConfMat, Rep, roc_auc=GetReport(model, PlotROC, X_test, y_test)
        
         
    return model, accur, ConfMat, Rep, roc_auc


#Perform a gridsearch on a particular model, print reports
def GridSearchModel(features, labels, model=LogisticRegression(penalty='l2', class_weight='auto'),  parameters = {'C':[ .01, .1, 10]}, Reports=False, PlotROC=False):
    
    
    accur=None
    ConfMat=None
    Rep=None
    roc_auc=None
    
    Bin_to_Int=np.vectorize(int)
    #Switch labels so we maximize recall of defaults
    def SwitchLabels(x):
        return (Bin_to_Int(x,2)+1)%2

    clf = grid_search.GridSearchCV(model, parameters, scoring="recall", refit=True)
    clf.fit(features, SwitchLabels(labels))
    print clf.best_estimator_
    
    if Reports:
        accur, ConfMat, Rep, roc_auc=GetReport(clf, PlotROC, features, SwitchLabels(labels))
        
         
    return clf, accur, ConfMat, Rep, roc_auc

#Function to plot learning curves
def PlotLearningCurves(model, features, labels):
    
    
    m, train_scores, test_scores = learning_curve(LogisticRegression(penalty='l2', class_weight='auto', C=10), features, labels, cv=None, n_jobs=1, train_sizes=np.arange(400, 10000, 100))
    Jtrain = np.mean(train_scores, axis=1)
    Jcv = np.mean(test_scores, axis=1)

    
    plt.figure(1)
    plt.clf()
    plt.plot(m,1.-Jtrain,'r',label='Training Error')
    plt.plot(m,1.-Jcv,'k',label='Cross Validation Error')
    plt.xlabel('Number of Training Points', fontsize=20)
    plt.ylabel('Error', fontsize=20)
    plt.title('Learning Curves for Logistic Regression Model', fontsize=20)
    plt.legend( loc='best', prop={'size':15})  
    plt.show()
    
    return 


#Backtest the model on Tier 4 loans
def BackTest(Feats, df, GSCV=False):
    
    models=Feats.keys()
    conditions=(df.IntCat=='Tier 4')
    ModNames=[]
    count=1
    
    def ConvBin(x):
        if GSCV:
            if x==0:
                return bin(1)
            else:
                return bin(0)
        else:
            return x
    
    
    for model in models:
        features, _, ind =GetFeatures(df, Feats[model][1], conditions, Feats[model][2])
        df[model]=(pd.Series(Feats[model][0].predict(features), index=ind)).apply(ConvBin)
        ModNames.append(model)
        count+=1
    
    features=None


    #Before considering any models
    grouped=df.dropna().groupby(['loan_status'])
    datbef=np.append(grouped.mean()[['PerLoss', 'Loss']].ix['0b0'].as_matrix(), 100.*(grouped.count()[['PerLoss']].ix['0b0']/float(df.dropna().PerLoss.count())).as_matrix())

    datmodels=[datbef]
    
    for name in ModNames:
        grouped=df.dropna().groupby([ name, 'loan_status'])
        datmodels.append(np.append(grouped.mean()[['PerLoss', 'Loss']].ix['0b1'].ix['0b0'].as_matrix(), 100.*(grouped.count()[['PerLoss']].ix['0b1'].ix['0b0']/float(df.dropna().groupby([name]).count().PerLoss.ix['0b1'])).as_matrix()))

    grouped=df.groupby(['loan_status'])
    dr=grouped.count().loan_amnt['0b0']/float(grouped.count().loan_amnt.sum())
    AvInt=df.int_rate.mean()
    Total=df.count().loan_amnt
    
    Full=[np.array([100.*dr,100.*AvInt, 100.*(AvInt-dr), np.nan])]
    
    for name in ModNames:
        
        cond=(df.IntCat!='Tier 4') | (df[name]!='0b0')
        grouped=df[cond].groupby(['loan_status'])
        dr=grouped.count().loan_amnt['0b0']/float(grouped.count().loan_amnt.sum())
        AvInt=df[cond].int_rate.mean()
        Totaltemp=df[cond].count().loan_amnt
        Full.append(np.array([100.*dr,100.*AvInt, 100.*(AvInt-dr), 100.0*(Total-Totaltemp)/float(Total)]))



   
    return pd.DataFrame(datmodels, columns=['AvePerLoss','AveLoss','DefaultRate'], index=['Before']+(ModNames)), pd.DataFrame(Full, columns=['DefaultRate','AverageInterest', 'AnticipatedReturn', 'ChangeInNumberofLoans'], index=['Before']+(ModNames))



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Begin main part of the program
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#Define data location and create a disk engine for SQL database
file1='.Data/LoanStats3a.csv'
file2='./Data/LoanStats3b.csv'
file3='./Data/LoanStats3c.csv'
file4='./Data/LoanStats3d.csv'
files=[file1,file2,file3,file4]

#Interesting Columns
cols=[ 'loan_amnt', 'annual_inc', 'int_rate', 'emp_length', 'home_ownership', 'earliest_cr_line', 'revol_util', 'loan_status', 'term', 'total_pymnt_inv',  'funded_amnt_inv']

#Import Data
df=CleanData(GetData(cols, files))

#Summary of defaulted loans
print pd.Series([100.*df.groupby('loan_status').count().loan_amnt.ix['0b0']/float(df.loan_amnt.count()), df.funded_amnt_inv.sum()/10**9, df.groupby('loan_status').sum().Loss.ix['0b0']/10**6], index=['Default Percentage', 'Total Funded Loans (billions)', 'Total Lost Principal (millions'])

#Plot distribution of principal lossses
PlotLostPrin(df)

#Summary of loans by interest category
print pd.DataFrame([df.groupby('IntCat').mean().int_rate*100., 100.*df.groupby(['loan_status','IntCat']).count().int_rate.ix['0b0']/df.groupby('IntCat').count().int_rate, df.groupby(['loan_status','IntCat']).mean().Loss.ix['0b0']]).T

#Look at distribution of loans by interest rate category
PlotFractions(df)

#Visualize Data for 4th quartile
Visualize2(df)

#Get ready to fit classifier. First define features and labels
RelFeats=[ 'loan_amnt', 'annual_inc', 'int_rate', 'emp_length', 'home_ownership', 'earliest_cr_line', 'revol_util', 'term']

#Remove outlierts and non-completed loans
conditions=(df.annual_inc<200000) & (df.int_rate>.15) & (df.earliest_cr_line<500) & (df.earliest_cr_line>50) & (df.revol_util<1.0) & (df['loan_status']!=2) & (df.IntCat=='Tier 4')
features, labels, _=GetFeatures(df, RelFeats, conditions,  polyorder=1)
feats={}

#Uncomment to fit without gridsearch
'''
#Fit Logistic Regression Model
model1, accur, ConfMat, Rep, roc_auc = FitLogReg(features, labels, RelFeats,c=10, split=.2, Reports=True, PlotROC=False)
feats['Logit']=[model1, RelFeats,1]
#Fit SVM Model
model2, accur, ConfMat, Rep, roc_auc = FitSVMReg(features, labels, RelFeats, c=10, split=.2, Reports=True, PlotROC=True)
feats['SVM']=[model2, RelFeats,1]
r1,r2=BackTest(feats, df)
print r1
print r2

'''

#Use grid search on logistic regression and SVM to search for best regularization parameter C
clf, accur, ConfMat, Rep, roc_auc= GridSearchModel(features, labels, model=LogisticRegression(penalty='l2', class_weight='auto'),  parameters = {'C':[ 0.001,.01, .1, 10, 100]}, Reports=True, PlotROC=False)
feats['Logit']=[clf, RelFeats, 1]

clf2, accur, ConfMat, Rep, roc_auc= GridSearchModel(features, labels, model=SVC( class_weight='auto', probability=True),  parameters = {'C':[ 0.001,.01, .1, 10, 100]}, Reports=True, PlotROC=False)
feats['SVM']=[clf2, RelFeats, 1]

#Now backtest the above results
r1,r2=BackTest(feats, df, GSCV=True)
print r1
print r2


#Plot learning curves
PlotLearningCurves(LogisticRegression(penalty='l2', class_weight='auto', C=10), features, labels)
