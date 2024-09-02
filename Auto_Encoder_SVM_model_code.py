from CommonMethodUsedForSchedulingProcess import *
from sklearn.metrics import accuracy_score
from sklearn import svm
from keras.models import load_model
from IEEE_TSC_commentsMethod import *
SF_jobDatsetWithOutReq2=r'Dataset/VM_DataSet/jobDatsetWithOutReq2Files'
splitDataFiles=SF_jobDatsetWithOutReq2+'/method2/dataset'

clusteringResult=SF_jobDatsetWithOutReq2+'/method2'+'/ClustringResult'+'/result'

compian_jobIDandTimFramandAllFeatures=reaCsvFile(SF_jobDatsetWithOutReq2,'/method2/allComainJobUnqIwithallFeatures.csv',(0,1,2,3,4,5,6))
print(np.shape(compian_jobIDandTimFramandAllFeatures))
print(compian_jobIDandTimFramandAllFeatures[:1])

"""


#count the job time different

"""
#print(TheJobTimedistribution(selectColumn(compian_jobIDandTimFramandAllFeatures,0,1)))
morethan20Days,from7to20,from0to7=TheJobTimedistribution(selectColumn(compian_jobIDandTimFramandAllFeatures,1))
print('job More Than 20',morethan20Days/len(compian_jobIDandTimFramandAllFeatures)
     ,'\nbetween 7 to 20',from7to20/len(compian_jobIDandTimFramandAllFeatures)
        ,'\nbetween 0 to 7',from0to7/len(compian_jobIDandTimFramandAllFeatures)
     )


"""selecting the data column for split files"""
print(len(compian_jobIDandTimFramandAllFeatures))
JobIDandTimeDuration=[]
for jobIDandT in compian_jobIDandTimFramandAllFeatures :
    JobIDandTimeDuration.append(selectDataFromRangWithConsiderNon(jobIDandT,[0,1]))

print(len(JobIDandTimeDuration))

# the ids of training and validation dataset
TrainingIdDATasetMore20 = []
ValidationIdDATasetMore20 = []
TestingIdDATasetMore20 = []
TrainingIdDATasetMore20 = reaCsvFile(splitDataFiles + '/from20to31', '/JobIDandtimeDurationTraining', 0)
ValidationIdDATasetMore20 = reaCsvFile(splitDataFiles + '/from20to31', '/JobIDandtimeDurationVal', 0)
TestingIdDATasetMore20 = reaCsvFile(splitDataFiles + '/from20to31', '/JobIDandtimeDurationTest', 0)

TrainingIdDATaset7to20 = []
ValidationIdDATaset7to20 = []
TestingIdDATaset7to20 = []
TrainingIdDATaset7to20 = reaCsvFile(splitDataFiles + '/from7to20', '/JobIDandtimeDurationTraining', 0)
ValidationIdDATaset7to20 = reaCsvFile(splitDataFiles + '/from7to20', '/JobIDandtimeDurationVal', 0)
TestingIdDATaset7to20 = reaCsvFile(splitDataFiles + '/from7to20', '/JobIDandtimeDurationTest', 0)

TrainingIdDATaset0to7 = []
ValidationIdDATaset0to7 = []
TestingIdDATaset0to7 = []
TrainingIdDATaset0to7 = reaCsvFile(splitDataFiles + '/Upto7days', '/JobIDandtimeDurationTraining', 0)
ValidationIdDATaset0to7 = reaCsvFile(splitDataFiles + '/Upto7days', '/JobIDandtimeDurationVal', 0)
TestingIdDATaset0to7 = reaCsvFile(splitDataFiles + '/Upto7days', '/JobIDandtimeDurationTest', 0)

print(len(TrainingIdDATasetMore20), '+', len(ValidationIdDATasetMore20), '+', len(TestingIdDATasetMore20),
      '=', len(TrainingIdDATasetMore20) + len(ValidationIdDATasetMore20) + len(TestingIdDATasetMore20))

print(len(TrainingIdDATaset7to20), '+', len(ValidationIdDATaset7to20), '+', len(TestingIdDATaset7to20),
      '=', len(TrainingIdDATaset7to20) + len(ValidationIdDATaset7to20) + len(TestingIdDATaset7to20))

print(len(TrainingIdDATaset0to7[200:]), '+', len(ValidationIdDATaset0to7[100:]), '+', len(TestingIdDATaset0to7[40:]),
      '=', len(TrainingIdDATaset0to7[200:]) + len(ValidationIdDATaset0to7[100:]) + len(TestingIdDATaset0to7[40:]))


"""spilt dataset to training dataset and validation datset """
trainingJobI=[]
trainingJobI=TrainingIdDATasetMore20+TrainingIdDATaset7to20+TrainingIdDATaset0to7
print(len(trainingJobI))
validationJobI=[]
validationJobI=ValidationIdDATasetMore20+ValidationIdDATaset7to20+ValidationIdDATaset0to7
print(len(validationJobI))
testingJobI=[]
testingJobI=TestingIdDATasetMore20+TestingIdDATaset7to20+TestingIdDATaset0to7
print(len(testingJobI))


trainingAndsValidationIDs=trainingJobI+validationJobI

allDataSetJobIDs=trainingJobI+validationJobI+testingJobI


""""

Training K-mean model 

"""

"""#collect all dataset  data 
"""
allDataSetJobDataWithJobIDs=[]
for Jobid in allDataSetJobIDs:
    for JobDATA in compian_jobIDandTimFramandAllFeatures:
            if Jobid==JobDATA[0]:
                    allDataSetJobDataWithJobIDs.append(selectDataFromRangWithConsiderNon(JobDATA,(0,1,2,3,4,5,6)))


"""#collect all dataset  data and get of the jobId from the data 
"""
allDataSetJobDatawithOutJobIDs=[]
for Jobid in allDataSetJobIDs:
    for JobDATA in compian_jobIDandTimFramandAllFeatures:
            if Jobid==JobDATA[0]:
                    allDataSetJobDatawithOutJobIDs.append(selectDataFromRangWithConsiderNon(JobDATA,(1,2,3,4,5,6)))


"""#collect tuple of three element for k-mean cluster for all dataSet 
"""
dfForAllDataset=[]
xForAllDataset=selectColumn(allDataSetJobDatawithOutJobIDs,2)
yForAllDataset=selectColumn(allDataSetJobDatawithOutJobIDs,4)
zForAllDataset=selectColumn(allDataSetJobDatawithOutJobIDs,3)
#gForAllDataset=selectColumn(allDataSetJobDatawithOutJobIDs,4)
#hForAllDataset=selectColumn(allDataSetJobDatawithOutJobIDs,0)
#kForAllDataset=selectColumn(allDataSetJobDatawithOutJobIDs,5)

for (xxForAllDataset,yyForAllDataset) in zip(xForAllDataset,yForAllDataset):
    dfForAllDataset.append((xxForAllDataset,yyForAllDataset))


""" K-mean Model """
""" this part wes commneded as the results of original expermant was saves in file So the result will be read form that file """
# kmeansForAllDataset = KMeans(n_clusters=5)
# kmeansForAllDataset.fit(dfForAllDataset)
# labelForAllDataset = kmeansForAllDataset.predict(dfForAllDataset)
#
# countClassElements(labelForAllDataset)
#
# # plottingCluster(xForAllDataset, yForAllDataset, zForAllDataset, labelForAllDataset)
#
#
# kMeanclusterOutPutForAllDataset=ConcancateLableTodataAfretAutoPerdict(labelForAllDataset,0,allDataSetJobDataWithJobIDs)
kMeanclusterOutPutForAllDataset=reaCsvFile(clusteringResult,'/result4',(0,1,2,3,4,5,6,7))

"""" Prepreing dataset for Auto-encoder and SVM model """

# spilt job data based on utiliaztion
#print(kMeanclusterOutPutForAllDataset)
trainingJobI1=[]
validationJobI1=[]
testingJobI1=[]
class0=[]
class1=[]
class2=[]
for i in kMeanclusterOutPutForAllDataset:
    if i[7]==1:
        class0.append(i[0])
    elif i[7]==0:
        class1.append(i[0])
    elif i[7]==3:
        class2.append(i[0])
print(len(class0),len(class1),len(class2))
print(int(len(class0)*.7),int(len(class0)*.8))
trainingJobI1.extend(class0[:int(len(class0)*.7)])
validationJobI1.extend(class0[int(len(class0)*.7):int(len(class0)*.8)])
testingJobI1.extend(class0[int(len(class0)*.8):])


print(int(len(class1)*.7),int(len(class1)*.8))

trainingJobI1.extend(class1[:int(len(class1)*.7)])
validationJobI1.extend(class1[int(len(class1)*.7):int(len(class1)*.8)])
testingJobI1.extend(class1[int(len(class1)*.8):])


print(int(len(class2)*.7),int(len(class2)*.8))

trainingJobI1.extend(class2[:int(len(class2)*.7)])
validationJobI1.extend(class2[int(len(class2)*.7):int(len(class2)*.8)])
testingJobI1.extend(class2[int(len(class2)*.8):])

print(np.shape(trainingJobI1),np.shape(validationJobI1),np.shape(testingJobI1))

""" Writing the result in the file """

#write the cluster result at method2 floder->clustringResult-.result
# writeFinalResultDataToFile(clusteringResult+'\\result4',Job7featuresHeader,clusterOutPut)


#split labled dataset to training, validation and testing datasets
trainigDatasetWithKMeanLable,valDatasetWithKMeanLable,testingDatasetWithKMeanLable=splitLabledDatasetToTrainingValidationAndTestingDatasets(kMeanclusterOutPutForAllDataset,trainingJobI1,validationJobI1,testingJobI1)
""" the shape of dataset """
print(np.shape(trainigDatasetWithKMeanLable))
print(np.shape(valDatasetWithKMeanLable))
print(np.shape(testingDatasetWithKMeanLable))

"""The result of the above code 

(1583, 8)
(225, 8)
(459, 8)

"""

"""The dataset with K-mean lable """

trainingAndValdationDatasetWithKmeanLabels=trainigDatasetWithKMeanLable+valDatasetWithKMeanLable


"""#collect data from VMData (5mints) with jobID and add the label aray as y_test data set
"""

trainingAndValjob=selectColumn(trainingAndValdationDatasetWithKmeanLabels,0)
trainingAndValjobl=selectColumn(trainingAndValdationDatasetWithKmeanLabels,7)

#print(job)
print(SF_jobDatsetWithOutReq2+'/dataset')
trainingAndValVMDataWithJoID,trainingAndValVMDataLabel=collectingDataFromJobFileswithJobIDWithLabelArray(trainingAndValjob,SF_jobDatsetWithOutReq2+'/dataset'+'/',0.00022831,trainingAndValjobl)

print(np.shape(trainingAndValVMDataWithJoID))

# # perpare the data with take-off the job id from the data
x_train6=[]
for i in trainingAndValVMDataWithJoID:
    x_train6.append(selectDataFromRangWithConsiderNon(i,[1,3,4,5,6,7,8,9,10,11,14,15,16,17,18,19]))
print(x_train6[:1])


"""" split the data set"""


X_train6=[]
x_validation6=[]
X_train6=x_train6[:3000]
x_validation6=x_train6[3000:]

print(np.shape(X_train6))
print(np.shape(x_validation6))



"""
The empelmntation of training the model have saved in Training_AutoEnocder and Auto_Encoder_SVM_model_code Packages "


"""
"""load the model
"""
model7=load_model('Saved_models/encodersPartOfThemodelTheBestforCode5.h5')

"""extract featues nusing encoder part
"""
extracting3Featrures=model7.predict(x_train6)

# extracting3Featrures and VMDataLabel for train k-nearest algorithm
#ex=[]
#for i in extracting3Featrures:
   # ex.append(selectDataFromRangWithConsiderNon(i,[0,2]))
#print(np.shape(ex))
X_trainForNearestAlg=extracting3Featrures
Y_trainForNearestAlg=trainingAndValVMDataLabel

print(np.shape(X_trainForNearestAlg),np.shape(Y_trainForNearestAlg))

"""
SVM Model
"""

#create clisifier with SVM
C=2048
clf1 = svm.SVC(kernel='linear', C=C)
clf2 =svm.LinearSVC(C=C, max_iter=10000000)
clf3 =svm.SVC(kernel='rbf', gamma=4096, C=C)
clf4 =svm.SVC(kernel='poly', degree=3, gamma='auto', C=C)

"""" Classify using SVM model"""


#fit classifier with data from Autoeencoder(X_trainForNearestAlg) and label form k-means(Y_trainForNearestAlg)
clf3.fit(extracting3Featrures,Y_trainForNearestAlg)

""" Predicting scoor """

predictSVMForTraingAndValDataset=[]
predictSVMForTraingAndValDataset.clear()

predictSVMForTraingAndValDataset=clf3.predict(extracting3Featrures)
accuracy_score(predictSVMForTraingAndValDataset,Y_trainForNearestAlg)


print(predictSVMForTraingAndValDataset)
print(len(predictSVMForTraingAndValDataset))
countClassElements(predictSVMForTraingAndValDataset)


#Confusion Matrix

cm=confusion_matrix(predictSVMForTraingAndValDataset,Y_trainForNearestAlg,labels=[0,3,4])
sns.heatmap(cm, annot=True,
            cmap='Blues')
cm.T

""" new copy of dataset """

trainingAndValdationDatasetWithKmeanLabels1= copy.deepcopy(trainingAndValdationDatasetWithKmeanLabels)

#labeling 7 feature with label from 5 mint job that have been labeled by knn
FiveMintJobWithLabelForTraingAndValDataset= ConcancateLableTodataAfretAutoPerdict(predictSVMForTraingAndValDataset,0,trainingAndValVMDataWithJoID)
"""
The following part is handel the comment of the reviews of IEEE Trasaction on Servces Computing 

The comment 
"The paper does not correctly describe the historical usage data needed for the predictive scheduling process. Also, why the focus on CPU and RAM. These two metrics have been extensively treated tackled in the literature. The authors must revisit the problem of VM scheduling/placement with high-priority metrics.
"

"""
"""==============================================================================
Here the function that handel the comment 
"""
SVM_LablesForTraingData=predictSVMForTraingAndValDataset
proprety_metrices(SVM_LablesForTraingData)

knnOutputForTraningAndValDataset=[]
knnOutputForTraningAndValDataset=labeling7featurewithlabelfrom5mintjobthathavebeenlabeledby_knn (FiveMintJobWithLabelForTraingAndValDataset,trainingAndValdationDatasetWithKmeanLabels1)


""" Testing Data set """

#collect data from VMData (5mints) with jobID and add the label aray as y_test data set
jobIDsforTestingDataSet=selectColumn(testingDatasetWithKMeanLable,0)
joblabeljobforTestingDataSet=selectColumn(testingDatasetWithKMeanLable,7)
VMDataWithJoIDforTestingDataSet,VMDataLabelforTestingDataSet=collectingDataFromJobFileswithJobIDWithLabelArray(jobIDsforTestingDataSet,SF_jobDatsetWithOutReq2+"/dataset"+'/',0.00022831,joblabeljobforTestingDataSet)

# perpare the data with take-off the job id from the data
x_test6 = []
for i in VMDataWithJoIDforTestingDataSet:
    x_test6.append(selectDataFromRangWithConsiderNon(i, [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19]))


""" Extracting features using encoders """

#extracting 3 from second autoencoders
extracting3FeatruresTestingDataset=model7.predict(x_test6)
print(extracting3FeatruresTestingDataset[:10])
print(np.shape(extracting3FeatruresTestingDataset))
ex2=[]
for i in extracting3FeatruresTestingDataset:
    ex2.append(selectDataFromRangWithConsiderNon(i,[0,2]))
print(np.shape(ex2))

""" classify using SVM model """

predictSVMForTestDataSet=clf3.predict(extracting3FeatruresTestingDataset)
print(predictSVMForTestDataSet)

# concanecate the label that output from k-NN for 5 mint job


FiveMintJobWithLabelFroTestingDataSet = ConcancateLableTodataAfretAutoPerdict(predictSVMForTestDataSet, 0,
                                                                              VMDataWithJoIDforTestingDataSet)


""" new copy of testing dataset """

VMDataWithJoIDforTestingDataSet2= copy.deepcopy(testingDatasetWithKMeanLable)

knnOutputFortestingDataSet=[]
knnOutputFortestingDataSet.clear()


#labeling 7 feature with label from 5 mint job that have been labeled by knn


knnOutputFortestingDataSet=labeling7featurewithlabelfrom5mintjobthathavebeenlabeledby_knn (FiveMintJobWithLabelFroTestingDataSet,VMDataWithJoIDforTestingDataSet2)


""" Accurcy scoor ddor testing data set """

accuracy_score(selectColumn(testingDatasetWithKMeanLable,7),selectColumn(knnOutputFortestingDataSet,7))

#confusionMatix for testing dataset
cm6=confusion_matrix(selectColumn(knnOutputFortestingDataSet,7),selectColumn(testingDatasetWithKMeanLable,7),labels=[0,3,1])

accuracyReport(selectColumn(testingDatasetWithKMeanLable,7),selectColumn(knnOutputFortestingDataSet,7))