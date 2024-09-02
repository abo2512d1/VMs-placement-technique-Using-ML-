# from CommonMethodUsedForSchedulingProcess import *
from CommonDirectionAndViraibles import *
from schedual import *
from OurScheduling_algorithm import *
from job import *
machineIDs=getMachineID(DataSetWith7features,0.5095887345679012,0.5109583333333333)

#0.361976408,0.63045282


print(np.shape(machineIDs))

finalmachineIDs=list(dict.fromkeys(machineIDs))

print(np.shape(finalmachineIDs))

machinesAttr=getMachineAttribute(finalmachineIDs,machineAttributes)

print(np.shape(machinesAttr))


"""create schedule with all physcail machines  
"""
"""
Create a Google scheduling simmulation
"""
GoogleSchedul=[]
for (i,j) in zip(finalmachineIDs,machinesAttr):
        GoogleSchedul.append(Schedual(i,j[2],j[1]))

# clear the Google schedul first

clearJobFormMachine(GoogleSchedul)

#=========================
#add the job to the machine
#print(schedual1[1].getMachineID())
#print(len(schedual1))
for i in GoogleSchedul:
        #print(i.getMachineID())
        jobForiMachine=returnJobforMachines(i.getMachineID(),DataSetWith7features)
        #print(np.shape(jobForiMachine))
        for j in jobForiMachine:
                #print(j[0],j[1],j[2],j[3],j[4],j[5],j[6])
                #print('machine',i.ph.phID)
                i.job.append(Job(j[0],j[1],j[2],j[3],j[4],j[5],j[6],0))
        jobForiMachine.clear()
        #print('after adding',len(jobForiMachine))

for i in range(len(finalmachineIDs)):
    GoogleSchedul[i].showMachineCapacity()

"""Calculating reource utilization for Google simulations"""
cpuUtilizationgoogle=calculateCPUUtilization1(GoogleSchedul)

countNumberOfJobThatScheduled(GoogleSchedul)

""""
Starting the scheduling process 

"""
"""#Mack acopy of google schedul
"""
LeastLoadedSchedul=copy.deepcopy(GoogleSchedul)

"""# get the jobList to use it for scheduling
"""
arrivalJobsList=[]
_,_,_,arrivalJobsList=countandReturnTheJobsBasedonLabels(GoogleSchedul)

print(len(arrivalJobsList))
import numpy as np

# Set the seed for reproducibility (optional)
np.random.seed(42)

# Generate a list of 6944 random numbers between 0 and 400
random_numbers = np.random.randint(0, 401, size=len(arrivalJobsList))

""" write the lable of each job from the file has them --- alrody classify the jobs """
for job in arrivalJobsList:
    for lables,priority in zip(TheLabeledDatasetFor7FeaturesFromfile,random_numbers):
        if job.getjobID() == lables[0]:
            job.label = lables[9]
            job.priority = priority
            break
"""#delete all jobs from the new schedule
"""
clearJobFormMachine(LeastLoadedSchedul)

#sort the job basd on arrive time
# arrivalJobsList.sort(key=lambda x:x.jobCPUUsage, reverse=True)
# arrivalJobsList.sort(key=lambda x:x.jobStartTime)
# arrivalJobsList.sort(key=lambda x:x.priority, reverse=True)
# Sort by jobStartTime (ascending), then priority (descending), then jobCPUUsage (descending)
arrivalJobsList.sort(key=lambda x: (x.jobStartTime, -x.priorty, -x.jobCPUUsage))

"""
Scheduling algorithm that is in OurScheduling_algorithm package 

"""
schedulingIntLeastLoaded(LeastLoadedSchedul,arrivalJobsList,.90)

"""Resource utilization is calculated using Google scheduling algorithm"""
# cpuUtilizationV2=calculateCPUUtilization2(LeastLoadedSchedul)

"""#detected overloaded PMs after moving Vm from overloaded PMs
"""
count=0
SLAVs1=[]
migrastionCostList=[]
addressofOverload=[1]
while addressofOverload!=[]:
    cpuUtilizationStep1 = calculateCPUUtilization2(LeastLoadedSchedul)
    addressofOverload = []
    overleadedPMsthereshold=0.76
    addressofOverload=DetectingOverloadedPM(cpuUtilizationStep1,overleadedPMsthereshold)
    print(addressofOverload)
    SLA,migrationCost=MigratedFromOverloadedToUnderloaded(LeastLoadedSchedul,addressofOverload,0.75)

    SLAVs1.append(SLA)#more than the upper thereshold with 75% and in this case no PM has exxcedded 100% means no violations Skip this SLAvs
    migrastionCostList.append(migrationCost)

    count+=1
print("migration steps=",count)
    # # cpuUtilizationStep2=calculateCPUUtilization2(LeastLoadedSchedul)
    # #
    # # addressofOverload=DetectingOverloadedPM(cpuUtilizationStep2,overleadedPMsthereshold)
    # print(addressofOverload)


#detected underloadedloaded PMs
addressofUnderload=[]
addressofUnderload=DetectingUnderloadedPM(cpuUtilizationStep1,.35)
print(addressofUnderload)


x2= CalaulateEnergyConsumption2(cpuUtilizationStep1)
print("SLAvs",SLAVs1)
print("sum SLAvs",sum(SLAVs1[1]))
print("migrastion cost",migrastionCostList)
costOfMigration=[]
for migVM in migrastionCostList[1]:
    CPU_size, memory_size_mb = migVM

    costOfMigration.append(MigrationCost(memory_size_mb))
print("number of migrated VM", len(migrastionCostList))
print("costOfMigration", sum(costOfMigration))


#vertulization
dfArrListv2=[]
for sc in LeastLoadedSchedul:
    dfArrListv2.append(calculatePercentageOfClassesForeEachNode(sc,1))
#=================
# dfArrListv2=dfArrListv2

# convert list to pd
LesstLoadSchedulV2Dataset=pd.DataFrame(dfArrListv2,columns=['CPUClster1','MemClster1','CPUClster2','MemClster2','CPUClster3','MemClster3'])
LesstLoadSchedulV2Dataset *=100
# plt.stackplot(range(0,798),LesstLoadSchedulV2Dataset["CPUClster1"],LesstLoadSchedulV2Dataset["CPUClster2"],LesstLoadSchedulV2Dataset["CPUClster3"] ,colors=[ '#CD6839','#1E90FF','green'])
# plt.ylim([1, 100])
# plt.show()

import matplotlib.pyplot as plt

# Create subplots: 1 row, 2 columns
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Define new color palettes
colors = ['yellow', 'blue', 'green']  # Coral, Deep Purple, Olive Green

# Plotting the CPU clusters on the first subplot (left side)
axes[0].stackplot(
    range(0, 798),
    LesstLoadSchedulV2Dataset["CPUClster1"],
    LesstLoadSchedulV2Dataset["CPUClster2"],
    LesstLoadSchedulV2Dataset["CPUClster3"],
    colors=colors
)
axes[0].set_title('CPU Utilization')
axes[0].set_ylim([1, 100])
axes[0].set_xlabel('PM')
axes[0].set_ylabel('Utilization (%)')

# Plotting the Memory clusters on the second subplot (right side)
axes[1].stackplot(
    range(0, 798),
    LesstLoadSchedulV2Dataset["MemClster1"],
    LesstLoadSchedulV2Dataset["MemClster2"],
    LesstLoadSchedulV2Dataset["MemClster3"],
    colors=colors
)
axes[1].set_title('Memory Utilization')
axes[1].set_ylim([1, 100])
axes[1].set_xlabel('PM')
axes[1].set_ylabel('Utilization (%)')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()
