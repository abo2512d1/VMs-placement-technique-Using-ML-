# ====================
# read csv file and return as list
from PackagesAndDependencies import *
# from CommonDirectionAndViraibles import *


def reaCsvFile(f_dir, fileName, selcolomn):
    fileDataAsList = []
    fileDataAsList = (
        list(genfromtxt(f_dir + fileName, delimiter=",", skip_header=1, dtype=None, loose=False, usecols=selcolomn)))
    return fileDataAsList


# ===================
# add the physical machine id to the job data with 7 features
def addMachineIdTo7featuresData(oldData, newData):
    dataset = []
    for i in oldData:
        for j in newData:
            if i[0] == j[0]:
                print('t')
                i[8] = j[4]
                break
    return dataset


# =====================

def selectColumn(array, columnNum):  # select one colum fo table
    retArray = []
    for a in array:
        retArray.append(a[columnNum])
    return retArray


# ======================
# function that arrange data baed on the range and if the data has non will make it 0
def selectDataFromRangWithConsiderNon(OneRecord, Datarange):
    data = []
    for r in Datarange:
        if OneRecord[r] == None:
            data.append(0)
        data.append(OneRecord[r])

    return data


# ==========================
def writeFinalResultDataToFile(FileDierction, fheader, array6):
    f1 = open(FileDierction, 'w')  # create or opne file for all jobs alone
    writer3 = csv.writer(f1)  # write the data to the file
    writer3.writerow(fheader)
    for i in range(len(array6)):
        writer3.writerow(array6[i])


# ==========================
def getMachineID(dataset, start_time, end_time):
    machineIDs = []
    for i in dataset:
        if i[6] < end_time and i[7] > start_time:
            machineIDs.append(i[1])
    return machineIDs


# =========================
def getMachineAttribute(machinesID, MachinesData):
    machineAttr = []
    for i in machinesID:
        for j in MachinesData:
            if i == j[0]:
                machineAttr.append(j)
                break
    return machineAttr


# =========================
# add the job to the machine
def returnJobforMachines(machineIDs, DataSet):
    dataToAdd = []
    avgCpu = 0
    avgMem = 0
    for i in DataSet:
        if i[1] == machineIDs:
            # if i[9]<0.363045282 and i[10]>0.361976408:
            avgCpu = (i[2] + i[4]) / 2
            avgMem = (i[3] + i[5]) / 2

            dataToAdd.append([i[0], avgCpu, avgMem, i[6], i[7], i[8], i[9]])
    return dataToAdd


# =========================
def calculateMaxOfUtalization(schedual):
    cpuMax = 0
    memoryMax = 0
    statrTime = 0
    endTime = 0
    duration = 0
    if len(schedual.job) > 0:
        if len(schedual.job) == 1:
            cpuMax = schedual.getjobCPUUsage()
            memoryMax = schedual.getjobMemUsage()
            statrTime = schedual.getjobStartTime()
            endTime = schedual.getjobEndTime()
            duration = schedual.getjobDuration()
        elif len(schedual.job) > 1:
            cpuMax = sum(schedual.getjobCPUUsage())
            memoryMax = sum(schedual.getjobMemUsage())
            statrTime = max(schedual.getjobStartTime())
            endTime = max(schedual.getjobEndTime())
            duration = max(schedual.getjobDuration())
    else:
        cpuMax = 0
        memoryMax = 0
        statrTime = 0
        endTime = 0
        duration = 0
    return cpuMax, memoryMax, statrTime, endTime, duration


# =====================
def calculateMaxOfUtalizationEach5Mints(schedual):
    startTimeclo = 0.5095887345679012
    cpuMax = 0
    memoryMax = 0
    statrTime = 0
    endTime = 0
    duration = 0
    retList = []
    retList1 = []
    for i in range(100):
        # print(i,'->',len(schedual.job))
        x = i * 0.000114155 + startTimeclo
        for j in schedual.job:
            # print(j.jobStartTime,'<',x+0.000114155,' and', j.jobEndTime,'>',x)
            if (j.jobStartTime < x and j.jobEndTime > 0.5109583333333333):
                # print('here')
                retList.append([j.jobCPUUsage / 9, j.jobMemUsage / 9, j.jobStartTime / 9,
                                j.jobEndTime / 9, j.jobDuration / 9])
                # print(retList)
            elif (j.jobStartTime < x + 0.000114155 and j.jobEndTime > x and (j.jobStartTime > x)):
                retList.append((j.jobCPUUsage, j.jobMemUsage, j.jobStartTime,
                                j.jobEndTime, j.jobDuration))

            # else:
            # print('f')
        retList1.append(retList)
        retList = []
    return retList1


# =====================
# for k,m in zip(schedual.getjobStartTime(),schedual.getjobEndTime()):
#                print('k',k,'x1',x,'m',m,'x2',x+0.000114155)
#                if (m>x and k<x+0.000114155):
#                        if len(schedual.job)>0:
#                            if len(schedual.job)==1:
#                                cpuMax=schedual.getjobCPUUsage()
#                                memoryMax=schedual.getjobMemUsage()
#                                statrTime=schedual.getjobStartTime()
#                                endTime=schedual.getjobEndTime()
#                                duration=schedual.getjobDuration()
#                            elif len(schedual.job)>1:
#                                cpuMax=sum(schedual.getjobCPUUsage())
#                                memoryMax=sum(schedual.getjobMemUsage())
#                                statrTime=max(schedual.getjobStartTime())
#                                endTime=max(schedual.getjobEndTime())
#                                duration=max(schedual.getjobDuration())
#                        else:
#                            cpuMax=0
#                            memoryMax=0
#                            statrTime=0
#                            endTime=0
#                            duration=0
# return cpuMax,memoryMax,statrTime,endTime,duration
# ====================
def calculateThePercentageOfUsage(pHCapacity, jobRequ, i):
    pec = 0
    # print('clcuate pe cent ',pHCapacity)
    pec = np.asarray(jobRequ) / np.asarray(pHCapacity)
    return pec


# ====================
def plot_distribution(inp):
    plt.figure()
    ax = sns.distplot(inp)
    plt.axvline(np.mean(inp), color="k", linestyle="dashed", linewidth=5)
    plt.axvline(stdev(inp), color='r', linestyle='-')
    plt.text(stdev(inp), .9, "SD: {:.4f}".format(stdev(inp)), color="r")
    _, max_ = plt.ylim()
    plt.text(
        np.mean(inp) + np.mean(inp) / 10,
        max_ - max_ / 10,
        "Mean: {:.4f}".format(np.mean(inp)),
    )

    return plt.figure


def plottingSD(x1, y1, c, t):
    fig, ax = plt.subplots()
    plt.scatter(x1, y1, color=c)
    plt.title(t)
    plt.xlabel("utiliazation")
    plt.ylabel("time")
    # ax = sns.distplot(selectColumn(SortCluster0,2))
    plt.axvline(np.mean(x1), color="k", linestyle="dashed", linewidth=5, label='mean')
    plt.text(np.mean(x1), .5, "Mean: {:.4f}".format(np.mean(x1)), color="r")
    # plt.scatter(selectColumn(SortCluster0,9),selectColumn(SortCluster0,4))
    plt.axvline(stdev(x1), color='r', linestyle='-')
    plt.text(stdev(x1), .9, "SD: {:.4f}".format(stdev(x1)), color="r")
    sns.displot(x=x1, y=y1)
    plt.title(t)
    plt.xlabel("utiliazation")
    plt.ylabel("time")
    return plt.figure



#collect data from job files (5 mint ) with JobID and create label array for each job element
def collectingDataFromJobFileswithJobIDWithLabelArray(jobsId,JobSourceFile,DiserTime,JobLabel):
    x_train=[]
    x=[]
    label=[]
    sumTime=0
    for (job,jobL) in zip(jobsId,JobLabel):
        file = open(JobSourceFile+str(job))
        reader = csv.reader(file)
        lines= len(list(reader))
        #print(reader)
        if lines>4:
            #print(job)
            dtat_resource_for_jobsList =list(genfromtxt(JobSourceFile+str(job), delimiter=",",skip_header = 1,dtype=None,loose=False))
            sumTime=0
                #jobsIDList=selectColumn(dtat_resource_for_jobsList,0)
                #print(len(dtat_resource_for_jobsList))
            for record in dtat_resource_for_jobsList:

                sumTime=sumTime+record[27]
                if sumTime<=DiserTime:
                #jobfile=np.array(genfromtxt(JobSourceFiles1+str(job[0])+str(job[1]), delimiter=",",skip_header = 1,dtype=None))
                #jobfile1=jobfile[0][:20]
                #if jobfile.size==2:
                #print(record)
                #print('T')
                #for record in jobfile:
                        x.extend(selectDataFromRangWithConsiderNon(record,[0,5,6,7,8,9,10,11,12,13,14,15,16,17,18,21,23,24,25,26,27]))
                        x_train.append(x)
                        x=[]
                        label.append(jobL)
                            #print('append')
                                    #if np.shape(record)[0]>0:
                                        #x_train=record[4:]
                else:
                        break

    return x_train,label


# reshape dataset

def ArrayReshape(JobData):
    arr = []
    for ite in JobData:
        arr.append(list(ite))

    return arr


#calculating Energy consumptions

def energyConsumptuin(cpuUt):
    engConsumption = []
    for i in cpuUt:
        if i == 0:
            engConsumption.append(10)
        elif i > 0 and i <= .1:
            engConsumption.append(93.7)

        elif i >= .1 and i < .20:
            engConsumption.append(97)
        elif i >= .20 and i < .30:
            engConsumption.append(101)
        elif i >= .30 and i < .40:
            engConsumption.append(105)

        elif i >= .40 and i < .50:
            engConsumption.append(110)

        elif i >= .50 and i < .60:
            engConsumption.append(116)

        elif i >= .60 and i < .70:
            engConsumption.append(121)

        elif i >= .70 and i < .80:
            engConsumption.append(125)

        elif i >= .80 and i < .90:
            engConsumption.append(129)

        elif i >= .90 and i < .1:
            engConsumption.append(133)

        elif i == 1:
            engConsumption.append(135)
        elif i > 1:
            engConsumption.append(145 * i)

    return engConsumption


""""
Common Methods for scheduling process 

"""


def calculaeUtilization(schedual):
    jobList = []
    stratTimeForScerual = 0.5095887345679012
    endTimeForScerual = 0.5109583333333333
    intervalTime = 0.000114155
    timeSlice = []
    for t in range(12):
        stratTimeForScerual + intervalTime
        timeSlice.append(stratTimeForScerual + (intervalTime * t))
    timeSlice.append(0.5109583333333333)
    # jobList=schedualgetjobCPUUsage()
    # for i in jobList:
    # print(timeSlice)
    # for k in range (12):
    # print(timeSlice[k+1]-timeSlice[k])
    CPUutilization = []
    priodCPUUtilization = 0
    for i in range(12):
        # print(i)
        priodCPUUtilization = 0
        for j in schedual.job:
            if (j.getStartTime() >= timeSlice[i] and j.getStartTime() < timeSlice[i + 1]) or (
                    j.getEndTime() > timeSlice[i]):
                # print("start_slice",timeSlice[i],"startTime=",j.getStartTime(),'ene_slice',timeSlice[i+1],"before",priodCPUUtilization)
                priodCPUUtilization = priodCPUUtilization + calculateThePercentageOfUsage(schedual.getPhCPUCap(),
                                                                                          j.getCPU(), 0)
                # print("after",priodCPUUtilization)
        CPUutilization.append(priodCPUUtilization)
    return CPUutilization


# ===============================
def calculateCPUUtilization1(schedul):
    cpuUtilizatioArrayforEachPM = []
    count = 0
    overallUtil = []
    for i in schedul:
        cpuUtilizatioArrayforEachPM.append(calculaeUtilization(i))

    for i, j in zip(cpuUtilizatioArrayforEachPM, range(len(cpuUtilizatioArrayforEachPM))):
        overallUtil.append(sum(i) / len(i))
        # print('PM',j,'=',sum(i)/len(i))
    for i in overallUtil:
        if i > 0:
            count = count + 1
    # print('overall utilzation for all PM=',sum(overallUtil)/len(overallUtil))
    return overallUtil


# ===================================
def calculateCPUUtilization2(schedul):
    cpuUtilizatioArrayforEachPM = []
    count = 0
    overallUtil = []
    for i in schedul:
        cpuUtilizatioArrayforEachPM.append(calculaeUtilization(i))

    for i, j in zip(cpuUtilizatioArrayforEachPM, range(len(cpuUtilizatioArrayforEachPM))):
        overallUtil.append(sum(i) / len(i))
        print('PM', j, '=', sum(i) / len(i))
    for i in overallUtil:
        if i > 0:
            count = count + 1
    print('overall utilzation for all PM=', sum(overallUtil) / count)
    return overallUtil


# =========================================
def CalaulateEnergyConsumption2(cpuUtil):
    EnergyConsumption = energyConsumptuin(cpuUtil)
    print(EnergyConsumption, sum(EnergyConsumption))
    return cpuUtil


# =========================================
def calculatePercentageOfClassesForeEachNode(schedulNode, threshold):
    class1CPU = 0
    class2CPU = 0
    class3CPU = 0
    class1Mem = 0
    class2Mem = 0
    class3Mem = 0
    machinecapacityCPU = schedulNode.getPhCPUCap()  # *threshold
    machinecapacityMem = schedulNode.getPhMemory()  # *threshold
    for j in schedulNode.job:
        if len(schedulNode.job) > 0:
            # print(j.getLabel())
            if j.getLabel() == 0:
                # print("class 1")
                class1CPU += j.getCPU()
                class1Mem += j.getMemory()
            elif j.getLabel() == 1:
                # print("class 2")
                class2CPU += j.getCPU()
                class2Mem += j.getMemory()
            else:
                # print("class 3")
                class3CPU += j.getCPU()
                class3Mem += j.getMemory()

            # totalCPUU+=j.getCPU()
            # totalMemU+=j.getMemory()

    return class1CPU / machinecapacityCPU, class1Mem / machinecapacityMem, class2CPU / machinecapacityCPU, class1Mem / machinecapacityMem, class3CPU / machinecapacityCPU, class3Mem / machinecapacityMem


def ConcancateLableTodataAfretAutoPerdict(outPutOfAuto, indexoflabelOfOutputAuto, datawithID):
    count = 0
    l = []
    l = datawithID
    retList = []
    for (x, y) in zip(l, outPutOfAuto):
        count = count + 1
        # print( x,y.item())
        # x[indexoflabelOfOrgenalData]=y[indexoflabelOfOutputAuto]
        retList.append(x + [y.item()])

    print(count)
    return retList


# ===========================================
def labeling7featurewithlabelfrom5mintjobthathavebeenlabeledby_SVM(FiveMin, VMData7):
    arr2 = []
    fiveMintLabledWithknn = copy.deepcopy(FiveMin)
    all7featuresDataLabledByKmeans = copy.deepcopy(VMData7)
    a = []
    x = 0
    for t7 in all7featuresDataLabledByKmeans:
        for t5 in fiveMintLabledWithknn:

            if t7[0] == t5[0]:
                a.append(t5[21])
                x = most_frequent(a)
        print('kmeanLbale', t7[9], 'SVM label', x)
        t7[9] = x
        arr2.append(t7)
        a.clear()
        x = 0
        # print('after append ',arr2[7])

    return arr2


# ==========================================
def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i

    return num


# ========================================
def DetectingOverloadedPM(UtilizationListForschedul, value):
    count = 0
    address = []
    print('the maximim Vlaue is', max(UtilizationListForschedul))
    print('the minimum Vlaue is', min(UtilizationListForschedul))
    x = len(UtilizationListForschedul)
    for i, j in zip(UtilizationListForschedul, range(x)):
        if i > value:
            # print(i)
            count = count + 1
            address.append(j)
    print('the count is', count)
    return address


# ========================================
def DetectingUnderloadedPM(UtilizationListForschedul, value):
    count = 0
    address = []
    print('the maximim Vlaue is', max(UtilizationListForschedul))
    print('the minimum Vlaue is', min(UtilizationListForschedul))
    x = len(UtilizationListForschedul)
    for i, j in zip(UtilizationListForschedul, range(x)):
        if i < value and i != 0:
            # print(i)
            count = count + 1
            address.append(j)
    print('the count is', count)
    return address


# ========================================
def wirteThelableFromtheresultofOurMOdelToTheSchedul(scheduljoblist, ML_resultList):
    for i in scheduljoblist:
        for j in ML_resultList:
            if i.jobID == j[0]:
                i.label = j[9]
                print('the ID', i.jobID, '<==>', j[0], 'the lable', i.label, '<----', j[9])


# ========================================
# count of jobs that are scheduled

def countNumberOfJobThatScheduled(schedul):
    count = 0
    for machine in schedul:
        count += len(machine.job)
    return count


# count the jobs based on labels
def countandReturnTheJobsBasedonLabels(schedalList):
    co0 = 0
    co1 = 0
    co2 = 0
    jobL = []

    for i in schedalList:
        for j in i.job:
            if j.label == 0:
                co0 = co0 + 1
                jobL.append(j)
            if j.label == 1:
                co1 = co1 + 1
                jobL.append(j)
            if j.label == 3:
                co2 = co2 + 1
            jobL.append(j)
    return co0, co1, co2, jobL


# ============================
# clear of schedual jobs from machine
def clearJobFormMachine(schedalList):
    for i in schedalList:
        i.job.clear()


# ==============================
# printing list of schedual jobs

def printMachinesJobs(schedalList):
    for i in schedalList:
        for j in i.job:
            print(j.printJob())


# ==============================
# printing list of schedual macjine

def printMachinesmachines(schedalList):
    for i in schedalList:
        print(i.showMachineCapacity())


# ==============================
# clusters total need of resources
def calculateNeedFoResources(l1):
    cpuR = 0
    MemR = 0
    for i in (l1):
        cpuR = cpuR + i.jobCPUUsage
        MemR = MemR + i.jobMemUsage
    return cpuR, MemR


# =============================
# calculate over all capacity ofor machine
def calacualteOverAllCapacity(machine):
    overallCPUCap = 0
    overallMemoryCap = 0
    overallCPUCap = sum(machine.getjobCPUUsage()) / machine.getPhCPUCap()
    overallMemoryCap = sum(machine.getjobMemUsage()) / machine.getPhMemory()
    return overallCPUCap, overallMemoryCap


# ============================
# return the per cent for all clusters in the machine

def calcullateClusterpercent(machine):
    clus0 = 0
    clus1 = 0  # cpu intensive and less number of jobs
    clus2 = 0
    sum1 = 0
    for j in machine.job:
        if j.label == 1:
            clus0 = clus0 + 1
        if j.label == 0:
            clus1 = clus1 + 1
        if j.label == 3:
            clus2 = clus2 + 1
    sum1 = clus0 + clus1 + clus2
    if sum1 == 0:
        sum1 = 1
    return clus0 / sum1, clus1 / sum1, clus2 / sum1


# ==============================
# calculate the capacity of machine plus the add job
def calacualteOverAllCapacityPlusAddedJob(machine, job1):
    overallCPUCap = 0
    overallMemoryCap = 0
    overallCPUCap = (sum(machine.getjobCPUUsage()) + job1.jobCPUUsage) / machine.getPhCPUCap()
    overallMemoryCap = (sum(machine.getjobMemUsage()) + job1.jobMemUsage) / machine.getPhMemory()
    return overallCPUCap, overallMemoryCap


# =============================
# calculte Utilization by schedual
def calculteUtilizationOfMachines(schedalList):
    utilizatioEach5Mint = []
    sumUtilazation2 = []
    s = []
    s1 = []
    counthostbeign100 = 0
    for i in schedalList:
        cpu, _, _, _, _ = calculateMaxOfUtalization(i)
        # print(i.ph.phCPUCap,cpu)
        if calculateThePercentageOfUsage(i.ph.phCPUCap, cpu, 0) <= 1:
            s = calculateThePercentageOfUsage(i.ph.phCPUCap, cpu, 0)
            sumUtilazation2.append(s)
            # print('machine',i.ph.phID,' the cput utlization %=',s)
        else:
            # print('machine',i.ph.phID)
            utilizatioEach5Mint = calculateMaxOfUtalizationEach5Mints(i)
            for j in utilizatioEach5Mint:
                s1 = calculateThePercentageOfUsage(i.ph.phCPUCap, j[0], 0)
                sumUtilazation2.append(s1[0].item())
                counthostbeign100 = counthostbeign100 + 1
                # print('FiveMint machine',i.ph.phID,' the cput utlization %=',s1[0].item())
    return sumUtilazation2, utilizatioEach5Mint, counthostbeign100


# ===============================
# calculate CPU capacity for machine
def countCPUandMemoryCapacityForMachine(scheduali):
    cpuU = []
    memeryU = []
    for sc in scheduali:
        x, y = calacualteOverAllCapacity(sc)
        if x != 0:
            cpuU.append(x)
        if y != 0:
            memeryU.append(y)
    # print(cpuU)
    # print(memeryU)
    countUnderCpu30 = 0
    countmore30Cpu = 0

    for i in cpuU:
        if i < .30:
            countUnderCpu30 = countUnderCpu30 + 1
        else:
            countmore30Cpu = countmore30Cpu + 1
    # print(countmore30Cpu,countUnderCpu30)

    countUnderMem30 = 0
    countmore30Mem = 0
    for i in memeryU:
        if i < .30:
            countUnderMem30 = countUnderMem30 + 1
        else:
            countmore30Mem = countmore30Mem + 1
    # print(countmore30Mem,countUnderMem30)


# =====================================
# list ceate lists 1-overloaded machine 2-underloaded machine

def OverloadedUnderloadedMachines(schedulList):
    overloadedl = []
    underloadedl = []
    normalLoadl = []
    emptymachinel = []
    for sc in schedulList:
        cpu, memory = calacualteOverAllCapacity(sc)
        # print(cpu)
        if cpu > 0.70:
            # print(sc.ph.phID)
            overloadedl.append(sc)
        elif cpu >= 0.30 and cpu <= 0.7:
            normalLoadl.append(sc)

        elif cpu > 0 and cpu < 0.3:
            underloadedl.append(sc)

        if cpu == 0:
            emptymachinel.append(sc)
    return overloadedl, underloadedl, normalLoadl, emptymachinel


# =====================================
# list ceate lists 1-overloaded machine 2-underloaded machine

def OverloadedUnderloadedMachines2(schedulList, uperThershold, lowerThershold):
    overloadedl = []
    underloadedl = []
    normalLoadl = []
    emptymachinel = []
    for sc in schedulList:
        cpu, memory = calacualteOverAllCapacity(sc)
        # print(cpu)
        if cpu > uperThershold:
            # print(sc.ph.phID)
            overloadedl.append(sc)
        elif cpu >= lowerThershold and cpu <= uperThershold:
            normalLoadl.append(sc)

        elif cpu > 0 and cpu < lowerThershold:
            underloadedl.append(sc)

        if cpu == 0:
            emptymachinel.append(sc)
    return overloadedl, underloadedl, normalLoadl, emptymachinel


# ====================================
# the schedual
def schedul(schedualList, jobList):
    rejectedJob = []
    # print('++++++++++++++++++',len(rejectedJob))
    count = 0
    for selectedJobForSch in jobList:
        count = 0
        cpuUse = 0
        memeryUse = 0
        cluster0 = 0
        cluster1 = 0
        cluster2 = 0
        jobLabel = 0
        rejectedJob.clear()
        for schedual in schedualList:
            cpuUse, memeryUse = calacualteOverAllCapacity(schedual)
            # if (schedual.getMachineID()==1715041632.0):
            # print("==1",schedual.getMachineID(),cpuUse,memeryUse)
            if (cpuUse < 0.70 and memeryUse < 0.70):
                cluster0, cluster1, cluster2 = calcullateClusterpercent(schedual)
                jobLabel = selectedJobForSch.getLabel()
                if (jobLabel == 0 and cluster0 < 0.4) or (jobLabel == 3 and cluster1 < 0.2) or (
                        jobLabel == 4 and cluster2 < 0.4):
                    cpuUsePlusJob, memeryUsePlusJob = calacualteOverAllCapacityPlusAddedJob(schedual, selectedJobForSch)
                    # if (schedual.getMachineID()==1715041632.0):
                    # print("==2",schedual.getMachineID(),cpuUsePlusJob,memeryUsePlusJob)
                    if (cpuUsePlusJob < 0.70 and memeryUsePlusJob < 0.70):
                        schedual.addJob(selectedJobForSch.getCPU(), selectedJobForSch.getMemory(),
                                        selectedJobForSch.getStartTime()
                                        , selectedJobForSch.getEndTime(), selectedJobForSch.getDuration(),
                                        selectedJobForSch.getLabel())
                        count = count + 1
                        # if (schedual.getMachineID()==1715041632.0):
                        # print("==3",schedual.getMachineID(),cpuUsePlusJob,memeryUsePlusJob)
                        break
        if count == 0:
            rejectedJob.append(selectedJobForSch)

    print(count)

    return rejectedJob


# ================================
# get the smallest job from schedual
def getTheSmallestJobFromSchedualList(schedualList):
    job = 0
    sc = copy.deepcopy(schedualList.job)
    sc.sort(key=lambda x: x.jobMemUsage)

    return sc[0], 0


# ================================
# moving jobs from overloaded machine to underloaded machine
# moving smallest jobs from overloaded to underloaded
def movingSmallestJobsFromOverloadedToUnderloaded(overLo, undLo):
    movedJobs = []
    SLAV = []
    CheckrejectJob = []
    rejectedJobList = []
    cpuCapacityDuringMigration = []
    for sch in overLo:
        CheckrejectJob.clear()
        job, index = getTheSmallestJobFromSchedualList(sch)
        CheckrejectJob = schedul(undLo, [job])  # schedul the job to normal machines
        # print('===============================>',len(rejectJob))
        if (len(CheckrejectJob)) == 0:
            movedJobs.append(job)  # put job in moved jobs list
            SLAV.append((sum(sch.getjobCPUUsage()) * .1) / job.getCPU())  # calculate SLAV
            cpuCapacityDuringMigration.append(sch.getPhCPUCap())
            # print('heeeeeer',sch.getPhCPUCap())
            sch.deletJob(index)  # delete selected job

        if (len(CheckrejectJob)) > 0:
            # movedJobs.append(job)#put job in moved jobs list
            rejectedJobList.append(job)
            sch.deletJob(index)  # delete selected job
    # print('herrr',len(cpuCapacityDuringMigration))
    return movedJobs, rejectedJobList, SLAV, cpuCapacityDuringMigration


# =============================
# coundClusterInschedula
def coundClusterInschedula(schedulaLis):
    for sc in schedulaLis:
        print(calcullateClusterpercent(sc))


# ==============================
def UtilizationCPUandMemory(schdealList):
    cpuUt = []
    memoryUt = []
    for i in schdealList:
        x, y = calacualteOverAllCapacity(i)
        cpuUt.append(x)
        memoryUt.append(y)
    return cpuUt, memoryUt
"""
common methods for training the model 
"""

# function
# that
# arrab = nge
# dta
# baed
# on
# the
# range


def selectDataFromRang(OneRecord, Datarange):
    data = []
    for r in Datarange:
        data.append(OneRecord[r])

    return data


# =====================

def selectColumn(array, columnNum):  # select one colum fo table
    retArray = []
    for a in array:
        retArray.append(a[columnNum])
    return retArray


# ====================
# function that arrange data baed on the range and if the data has non will make it 0
def selectDataFromRangWithConsiderNon(OneRecord, Datarange):
    data = []
    for r in Datarange:
        if OneRecord[r] == None:
            data.append(0)
        data.append(OneRecord[r])

    return data


# ====================
# testin if the data set has none values
def testTensorArrrayIfItHasNone(arr):
    for job in arr:
        for i in job:
            if i == float('nan'):
                print('None')
            # else:
            # print(i)


# ====================
# read csv file and return as list

def reaCsvFile(f_dir, fileName, selcolomn):
    fileDataAsList = []
    fileDataAsList = (
        list(genfromtxt(f_dir + fileName, delimiter=",", skip_header=1, dtype=None, loose=False, usecols=selcolomn)))
    return fileDataAsList


# ===================
# read csv file and return as list with determin space
Dtype = [str, float, float, float, float, float, float, int]


def reaCsvFile1(f_dir, fileName):
    fileDataAsList = []
    fileDataAsList = (genfromtxt(f_dir + fileName, delimiter="	", skip_header=1, dtype=None, loose=False))
    return fileDataAsList


# ===================
def ConcancateLableTodataAfretAutoPerdict(outPutOfAuto, indexoflabelOfOutputAuto, datawithID):
    count = 0
    l = []
    l = datawithID
    retList = []
    for (x, y) in zip(l, outPutOfAuto):
        count = count + 1
        print(x, y.item())
        # x[indexoflabelOfOrgenalData]=y[indexoflabelOfOutputAuto]
        retList.append(x + [y.item()])

    print(count)
    return retList


# ====================
# write the predicted label to array alonge with job id and all featrurs in arry then in file array name is jobIDandTimFramandAllFeatures

def addTheperdictedlableToTrainAndValidatedataSetandCollectAllJobDatasets(datasetWithAllJobsAndAllfeatures,
                                                                          predictedLableWithJobID):
    returnArray4 = []
    for jobD in datasetWithAllJobsAndAllfeatures:
        if jobD[7] == -1:
            for PrJob in predictedLableWithJobID:

                if jobD[0] == PrJob[0]:
                    print(jobD[0], PrJob[0])
                    jobD[7] = PrJob[20]
                    print('J', jobD[7], 'lable', PrJob[20])
                    returnArray4.append(jobD)
                    break
        else:
            returnArray4.append(jobD)
    return returnArray4


# ==========================
def writeFinalResultDataToFile(FileDierction, fheader, array6):
    f1 = open(FileDierction, 'w')  # create or opne file for all jobs alone
    writer3 = csv.writer(f1)  # write the data to the file
    writer3.writerow(fheader)
    for i in range(len(array6)):
        writer3.writerow(array6[i])


# ===========================
# add label to all 7features and all data of jobs after clustering them by kmean

def addLabelto7FeaturesForAllJobs(vmData, KmeanLabels):
    arr = []
    Da = vmData
    count = 0
    for (Djob, l) in zip(Da, KmeanLabels):
        print('d', Djob, 'l', l.item())
        count = count + 1
        arr.append(Djob.append(l))  # .item())
    print(count)
    return arr


# ===========================
# write the predicted label to array alonge with job id and all featrurs in arry then in file array name is jobIDandTimFramandAllFeatures

def addTheperdictedlableToTestdataSetandCollectAllJobDatasets(datasetWithAllJobsAndAllfeatures,
                                                              predictedLableWithJobID):
    returnArray8 = []
    for jobD in datasetWithAllJobsAndAllfeatures:
        if jobD[7] == -1:
            for PrJob in predictedLableWithJobID:

                if jobD[0] == PrJob[0]:
                    print(jobD[0], PrJob[0])
                    jobD[7] = PrJob[20]
                    print('J', jobD[7], 'lable', PrJob[20])
                    returnArray8.append(jobD)
                    break
        else:
            returnArray8.append(jobD)
    return returnArray8


# ==========================
def writeFinalResultDataToFile(FileDierction, fheader, array8):
    f1 = open(FileDierction, 'w')  # create or opne file for all jobs alone
    writer3 = csv.writer(f1)  # write the data to the file
    writer3.writerow(fheader)
    for i in range(len(array8)):
        writer3.writerow(array8[i])


# ==========================
# confusion matrix plotting
import matplotlib.pyplot as plt2


def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap)  # imshow
    # plt.title(title)
    plt2.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt2.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt2.yticks(tick_marks, df_confusion.index)
    # plt.tight_layout()
    plt2.ylabel(df_confusion.index.name)
    plt2.xlabel(df_confusion.columns.name)


# =========================
# seperate jobs from one file to jobs files


def ReadJobsFromOneFileandWriteThemToFiles(JobsName, JobOneFileData, DiestinatioOfOutPutFiles, h):
    collecData = []
    for job in JobsName:
        for jobData in JobOneFileData:
            if jobData[0] == job:
                collecData.append(jobData)

        writeFinalResultDataToFile(DiestinatioOfOutPutFiles + str(job), h, collecData)
        collecData = []


# ===========================
# reshape dataset

def ArrayReshape(JobData):
    arr = []
    for ite in JobData:
        arr.append(list(ite))

    return arr


# =========================
# marrage tow array
def appendListAtTheEndOfAnotherList(lst1, lst2):
    arraY = []
    arraY.append(lst1)
    for i in lst2:
        arraY.extend(i)
    return arraY


# =========================
# count the job time different
def TheJobTimedistribution(JoTime):
    morthen20days = 0
    from7to20days = 0
    from0to7days = 0
    for jobT in JoTime:
        if jobT >= 0.657534:
            morthen20days = morthen20days + 1

        elif jobT < 0.657534 and jobT > 0.230137:
            from7to20days = from7to20days + 1
        else:
            from0to7days = from0to7days + 1
    return morthen20days, from7to20days, from0to7days


# ========================
def SpiltJobIdeToTrainingDatsetAndValDaset(NumberOfJobMorethan20Dayes, NumberOfJobfrom7to20Dayes,
                                           NumberOfJobfrom0to7Dayes, percentageOfTraingDataSet, percentageOfValeDataSet,
                                           JobIdAndTime):
    countJobMoreThan20t = 0
    countJobMoreThan20v = 0
    arrJobIDMoreThan20T = []
    arrJobIDMoreThan20V = []
    arrJobIDMoreThan20Te = []
    countJobfrom7to20t = 0
    countJobfrom7to20v = 0
    arrJobIDfrom7to20T = []
    arrJobIDfrom7to20V = []
    arrJobIDfrom7to20Te = []
    countJobFrom0to7t = 0
    countJobFrom0to7v = 0
    arrJobIDFrom0to7T = []
    arrJobIDFrom0to7V = []
    arrJobIDFrom0to7Te = []

    for job in JobIdAndTime:
        # print(job[1])
        if job[1] >= 0.657534:
            if len(arrJobIDMoreThan20T) / NumberOfJobMorethan20Dayes <= percentageOfTraingDataSet:
                countJobMoreThan20t = countJobMoreThan20t + 1
                arrJobIDMoreThan20T.append(job)
            elif len(arrJobIDMoreThan20V) / NumberOfJobMorethan20Dayes <= percentageOfValeDataSet:
                countJobMoreThan20v = countJobMoreThan20v + 1
                arrJobIDMoreThan20V.append(job)
            else:
                arrJobIDMoreThan20Te.append(job)


        elif job[1] < 0.657534 and job[1] > 0.230137:
            if len(arrJobIDfrom7to20T) / NumberOfJobfrom7to20Dayes <= percentageOfTraingDataSet:
                countJobfrom7to20t = countJobfrom7to20t + 1
                arrJobIDfrom7to20T.append(job)

            elif len(arrJobIDfrom7to20V) / NumberOfJobfrom7to20Dayes <= percentageOfValeDataSet:
                countJobfrom7to20v = countJobfrom7to20v + 1
                arrJobIDfrom7to20V.append(job)
            else:
                arrJobIDfrom7to20Te.append(job)
        else:
            if len(arrJobIDFrom0to7T) / NumberOfJobfrom0to7Dayes <= percentageOfTraingDataSet:
                countJobFrom0to7t = countJobFrom0to7t + 1
                arrJobIDFrom0to7T.append(job)
            elif len(arrJobIDFrom0to7V) / NumberOfJobfrom0to7Dayes <= percentageOfValeDataSet:
                countJobFrom0to7v = countJobFrom0to7v + 1
                arrJobIDFrom0to7V.append(job)
            else:

                arrJobIDFrom0to7Te.append(job)

        print('countJobMoreThan20t', countJobMoreThan20t, 'countJobMoreThan20v', countJobMoreThan20v)

    her = ['UniqJobId', 'timeDuration']
    writeFinalResultDataToFile(splitDataFiles + '\\from20to31\\JobIDandtimeDurationTraining', her, arrJobIDMoreThan20T)
    writeFinalResultDataToFile(splitDataFiles + '\\from20to31\\JobIDandtimeDurationVal', her, arrJobIDMoreThan20V)
    writeFinalResultDataToFile(splitDataFiles + '\\from20to31\\JobIDandtimeDurationTest', her, arrJobIDMoreThan20Te)

    writeFinalResultDataToFile(splitDataFiles + '\\from7to20\\JobIDandtimeDurationTraining', her, arrJobIDfrom7to20T)
    writeFinalResultDataToFile(splitDataFiles + '\\from7to20\\JobIDandtimeDurationVal', her, arrJobIDfrom7to20V)
    writeFinalResultDataToFile(splitDataFiles + '\\from7to20\\JobIDandtimeDurationTest', her, arrJobIDfrom7to20Te)

    writeFinalResultDataToFile(splitDataFiles + '\\Upto7days\\JobIDandtimeDurationTraining', her, arrJobIDFrom0to7T)
    writeFinalResultDataToFile(splitDataFiles + '\\Upto7days\\JobIDandtimeDurationVal', her, arrJobIDFrom0to7V)
    writeFinalResultDataToFile(splitDataFiles + '\\Upto7days\\JobIDandtimeDurationTest', her, arrJobIDFrom0to7Te)


# ==========================
# split the data to clusered and not clusterd data set
# collect data that has been clustered

def collectDataThatHasBeenClustered(Dataset3):
    returList3 = []
    for Dat in Dataset3:
        if Dat[7] != -1:
            returList3.append(Dat)
    return returList3


def collectDataThatHasNotBeenClustered(Dataset3):
    returList3 = []
    for Dat in Dataset3:
        if Dat[7] == -1:
            returList3.append(Dat)
    return returList3


# ===============================
def labeling7featurewithlabelfrom5mintjobthathavebeenlabeledby_knn(FiveMin, VMData7):
    arr2 = []
    fiveMintLabledWithknn = copy.deepcopy(FiveMin)
    all7featuresDataLabledByKmeans = copy.deepcopy(VMData7)
    a = []
    x = 0
    for t7 in all7featuresDataLabledByKmeans:
        for t5 in fiveMintLabledWithknn:

            if t7[0] == t5[0]:
                a.append(t5[21])
                x = most_frequent(a)
        print('kmeanLbale', t7[7], 'KnnLabel', x)
        t7[7] = x
        arr2.append(t7)
        a.clear()
        x = 0
        # print('after append ',arr2[7])

    return arr2


# ==============================
def accuracyReport(y_truth, y_predict):
    # precision_score(selectColumn(clusterOutPut[:1763],7), selectColumn(knnOutput,7), average=None)
    # precision_score(selectColumn(clusterOutPut[:1763],7), selectColumn(knnOutput,7), average='micro')
    # recall_score(selectColumn(clusterOutPut[:1763],7), selectColumn(knnOutput,7), average=None)
    target_names = ['class 0', 'class 1', 'class 2']
    print(classification_report(y_truth, y_predict, target_names=target_names))


# ================================
# plotting cluster
def plottingCluster(x, y, z, labell):
    ax1 = fig.gca(projection='3d')
    ax1.scatter(x, y, z, c=labell, s=50, alpha=0.4, edgecolors='w')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Max Usage of CPU')
    ax1.set_zlabel('Max Usage Memory')
    return ax1.figure


# collect data from job files (5 mint ) with JobID
def collectingDataFromJobFileswithJobID(jobsId, JobSourceFile, DiserTime):
    x_train = []
    x = []

    sumTime = 0
    for job in jobsId:
        file = open(JobSourceFile + str(job))
        reader = csv.reader(file)
        lines = len(list(reader))
        # print(reader)
        if lines > 4:
            print(job)
            dtat_resource_for_jobsList = list(
                genfromtxt(JobSourceFile + str(job), delimiter=",", skip_header=1, dtype=None, loose=False))
            sumTime = 0
            # jobsIDList=selectColumn(dtat_resource_for_jobsList,0)
            # print(len(dtat_resource_for_jobsList))
            for record in dtat_resource_for_jobsList:

                sumTime = sumTime + record[27]
                if sumTime <= DiserTime:
                    # jobfile=np.array(genfromtxt(JobSourceFiles1+str(job[0])+str(job[1]), delimiter=",",skip_header = 1,dtype=None))
                    # jobfile1=jobfile[0][:20]
                    # if jobfile.size==2:
                    # print(record)
                    # print('T')
                    # for record in jobfile:
                    x.extend(selectDataFromRangWithConsiderNon(record,
                                                               [0, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 21, 22,
                                                                23, 24, 25, 26, 27]))
                    x_train.append(x)
                    x = []

                    # print('append')
                    # if np.shape(record)[0]>0:
                    # x_train=record[4:]
                else:
                    break

    return x_train


# collect data from job files (5 mint ) with JobID and create label array for each job element
def collectingDataFromJobFileswithJobIDWithLabelArray(jobsId, JobSourceFile, DiserTime, JobLabel):
    x_train = []
    x = []
    label = []
    sumTime = 0
    for (job, jobL) in zip(jobsId, JobLabel):
        file = open(JobSourceFile + str(job))
        reader = csv.reader(file)
        lines = len(list(reader))
        # print(reader)
        if lines > 4:
            print(job)
            dtat_resource_for_jobsList = list(
                genfromtxt(JobSourceFile + str(job), delimiter=",", skip_header=1, dtype=None, loose=False))
            sumTime = 0
            # jobsIDList=selectColumn(dtat_resource_for_jobsList,0)
            # print(len(dtat_resource_for_jobsList))
            for record in dtat_resource_for_jobsList:

                sumTime = sumTime + record[27]
                if sumTime <= DiserTime:
                    # jobfile=np.array(genfromtxt(JobSourceFiles1+str(job[0])+str(job[1]), delimiter=",",skip_header = 1,dtype=None))
                    # jobfile1=jobfile[0][:20]
                    # if jobfile.size==2:
                    # print(record)
                    # print('T')
                    # for record in jobfile:
                    x.extend(selectDataFromRangWithConsiderNon(record,
                                                               [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                                                21, 23, 24, 25, 26, 27]))
                    x_train.append(x)
                    x = []
                    label.append(jobL)
                    # print('append')
                    # if np.shape(record)[0]>0:
                    # x_train=record[4:]
                else:
                    break

    return x_train, label

def MigrationCost(memory_size_mb):
    # Convert memory size from MB to Gb
    bandwidth_gbps=5 # means 5 GB
    memory_size_mb_to_gb = memory_size_mb / 1024  # Convert MB to GB
    memory_size_gb_to_gb = memory_size_mb_to_gb * 8  # Convert GB to Gb

    # Calculate transfer time in seconds
    transfer_time_seconds = memory_size_gb_to_gb / bandwidth_gbps

    # Cost per second is assumed to be 1 unit
    cost_per_second = 1

    # Calculate downtime cost
    downtime_cost = transfer_time_seconds * cost_per_second

    return downtime_cost

# ===============================
def splitLabledDatasetToTrainingValidationAndTestingDatasets(AllLabeleddataset, traininfIDs, valID, testIDs):
    triningDataset = []
    validationDataset = []
    testingDataset = []
    for i1 in traininfIDs:
        for j1 in AllLabeleddataset:
            if i1 == j1[0]:
                triningDataset.append(j1)

    for i2 in valID:
        for j2 in AllLabeleddataset:
            if i2 == j2[0]:
                validationDataset.append(j2)

    for i3 in testIDs:
        for j3 in AllLabeleddataset:
            if i3 == j3[0]:
                testingDataset.append(j3)

    return triningDataset, validationDataset, testingDataset


# ==================================
# Program to find most frequent
# element in a list

def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i

    return num


# ================================
# count classes elements
def countClassElements(lisst):
    count0 = 0
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    for i in lisst:
        if i == 0:
            count0 = count0 + 1
        elif i == 1:
            count1 = count1 + 1
        elif i == 2:
            count2 = count2 + 1
        elif i == 3:
            count3 = count3 + 1
        else:
            count4 = count4 + 1

    print('0=', count0, '1=', count1, '2=', count2, '3=', count3, '4=', count4)


# common methods for Autoencoder
def Plotting(his):
    # plotting the result
    plt.plot(his.history['accuracy'])
    plt.plot(his.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(his.history['loss'])
    plt.plot(his.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# =========================
# method that convert none filuse to zero

# np.nan_to_num(x_train, copy=False)

# =========================


