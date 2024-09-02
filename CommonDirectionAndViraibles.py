from CommonMethodUsedForSchedulingProcess import *

#common virables
dir=r'\\uniwa.uwa.edu.au\userhome\students8\22772098\My Documents\clustringProject\newDataseWithCPUDistribution'
dir2=r'\\uniwa.uwa.edu.au\userhome\students8\22772098\My Documents\clustringProject\schedulingDataSetAndSuorceFileForCode'

SF_jobDatsetWithOutReq2=dir+'\jobDatsetWithOutReq2Files'
dataset='\\dataset'
labelFiles='\labelFile'
KmeanFiles='\\filesForKemanData'
balncedData=dir+'\\jobDatsetWithOutReq2Files\\balanceTheData'
splitDataFiles=SF_jobDatsetWithOutReq2+'\\method2'+dataset
clusteringResult=SF_jobDatsetWithOutReq2+'\\method2'+'\\ClustringResult'+'\\result'
finalKNNresultWith7featreFileSource=SF_jobDatsetWithOutReq2+'\\method2'+'\\testingDatasetResult\\for7featuresResult'
SchedulingSourveFile=dir2
Job7featuresHeader=['UniqJobId','timeDuration','max_CPU','avg_CPU','max_memory','avg_memory','assigned_memory','label'
,'machine_id','start_time','end_time']
Job7featuresHeader2=['UniqJobId','machine_id','max_cpu','max_memory','avg_cpu','avg_memory','start_time','end_time','timeDuration','label']

knn_result=SF_jobDatsetWithOutReq2+'\\method2'+'\\knnResult'
distination=SF_jobDatsetWithOutReq2+'\\balanceTheData\\dataset\\'
#write in files
headerF=['UniqJobId','collection_id','instance_index','priority','machine_id','sample_rate',
         'cpu_usage_distribution1','cpu_usage_distribution2','cpu_usage_distribution3',
         'cpu_usage_distribution4','cpu_usage_distribution5','cpu_usage_distribution6',
         'cpu_usage_distribution7','cpu_usage_distribution8','cpu_usage_distribution9',
         'cpu_usage_distribution10','cpu_usage_distribution11','max_cpu','avg_cpu','cycles_per_instruction',
         'memory_accesses_per_instruction','assigned_memory','page_cache_memory','max_memory','avg_memory',
         'start_time','end_time','time_window']
distination=SF_jobDatsetWithOutReq2+'\\balanceTheData\\dataset\\'

''''

The follwing is the direction for training and testing the model 


'''
#common virables
dir=r'\\uniwa.uwa.edu.au\userhome\students8\22772098\My Documents\clustringProject\newDataseWithCPUDistribution'
SF_jobDatsetWithOutReq2=dir+'\jobDatsetWithOutReq2Files'
dataset='\\dataset'
labelFiles='\labelFile'
KmeanFiles='\\filesForKemanData'
balncedData='\\jobDatsetWithOutReq2Files\\balanceTheData'
splitDataFiles=SF_jobDatsetWithOutReq2+'\\method2'+dataset
clusteringResult=SF_jobDatsetWithOutReq2+'\\method2'+'\\ClustringResult'+'\\result'
Job7featuresHeader=['UniqJobId','timeDuration','max_CPU','avg_CPU','max_memory','avg_memory','assigned_memory','label'
]
knn_result=SF_jobDatsetWithOutReq2+'\\method2'+'\\knnResult'
#read from csv files

data_resource=dir+'\\jobDatsetWithOutReq2Files\\sourceFilesConsistsVMData\\jobDatsetWithOutReq2.csv'# the resource of job data
JobID_resource=dir+'\\jobWithTimeDuration.csv'


#clustring the balancing data
data_resourceForDataBalancing1=SF_jobDatsetWithOutReq2+'\\balanceTheData\\jobDatsetWithOutReq2_1.csv'
data_resourceForDataBalancing2=SF_jobDatsetWithOutReq2+'\\balanceTheData\\jobDatsetWithOutReq2_2_2.csv'
#balncingVMData1=list(genfromtxt(data_resourceForDataBalancing1, delimiter=",",skip_header = 1,dtype=None))
#balncingVMData2=list(genfromtxt(data_resourceForDataBalancing2, delimiter=",",skip_header = 1,dtype=None))
# balancingJob1=job2[594:594+586]
# balancingJob2=job2[594+586:]
# print(np.shape(balancingJob1),np.shape(balancingJob2))


#write in files
headerF=['UniqJobId','collection_id','instance_index','priority','machine_id','sample_rate',
         'cpu_usage_distribution1','cpu_usage_distribution2','cpu_usage_distribution3',
         'cpu_usage_distribution4','cpu_usage_distribution5','cpu_usage_distribution6',
         'cpu_usage_distribution7','cpu_usage_distribution8','cpu_usage_distribution9',
         'cpu_usage_distribution10','cpu_usage_distribution11','max_cpu','avg_cpu','cycles_per_instruction',
         'memory_accesses_per_instruction','assigned_memory','page_cache_memory','max_memory','avg_memory',
         'start_time','end_time','time_window']
distination=SF_jobDatsetWithOutReq2+'\\balanceTheData\\dataset\\'

distination=SF_jobDatsetWithOutReq2+'\\balanceTheData\\dataset\\'
# ReadJobsFromOneFileandWriteThemToFiles(balancingJob2,balncingVMData2,distination,headerF)
"""============================================================"""
"""++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"""
"""
The follwing is the direction used in the experemnt for scheduling part 

"""

#Migrated Data ste direction
MigratedDataSetDirection=r'Dataset/VM_DataSet/Google_AataSet_For_800PMs/newdatasetThatHas800PMs/dataset12hoursForDay15/datasetFor1HourInMiddelOFDay15/'#C:\Users\22772098\OneDrive - The University of Western Australia\My_Data_Set_Back_Up'
DataSetWith7features=reaCsvFile(MigratedDataSetDirection,'Collect7fetuesForJobIdes7802-2.csv',[0,1,2,3,4,5,6,7,8,9])
machineAttributes=reaCsvFile(MigratedDataSetDirection,'physcalMachineAttributeAndTime.csv',[0,1,2,3,4])
#Reading the knn result that has 7 features
#DataSetWith7features=reaCsvFile(MainDirectForFiles,'\Collect7fetuesForJobIdes7802-2.csv',[0,1,2,3,4,5,6,7,8,9])
#realKNNResult=reaCsvFile(SchedulingSourveFile+'\\dataset','\\resultKmeanToAutoencodersTaringDataValDataTestDatToKNN2',[0,1,2,3,4,5,6,7])
#machineAttributes=reaCsvFile(SchedulingSourveFile+'\\dataset','\\physcalMachineAttributeAndTime.csv',[0,1,2,3,4])


"""read the result from saved file 
"""
TheLabeledDatasetFor7FeaturesFromfile=reaCsvFile(MigratedDataSetDirection+'/ML_results','/OurModelResult',[0,1,2,3,4,5,6,7,8,9])
print(len(TheLabeledDatasetFor7FeaturesFromfile))