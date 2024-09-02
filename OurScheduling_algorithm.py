# v.2 of our algorithm
from CommonMethodUsedForSchedulingProcess import *
from migratedjobs import *
def schedulingIntLeastLoadedV2(macineList, jobList, threshold):
    for job in jobList:
        placeTheJob = 0
        schedulStatus = []
        hostMachine = []
        PercentClass1CPU = 0
        PercentClass1Mem = 0
        PercentClass2CPU = 0
        PercentClass2Mem = 0
        PercentClass3CPU = 0
        PercentClass3Mem = 0
        thersholForUtiClass1CPU = 0.55
        thersholForUtiClass1Mem = 0.20
        thersholForUtiClass2CPU = 0.25
        thersholForUtiClass2Mem = 0.25
        thersholForUtiClass3CPU = 0.20
        thersholForUtiClass3Mem = 0.55
        macineList.sort(key=lambda x: (x.getPhCPUCap() - (sum(x.getjobCPUUsage()))), reverse=True)
        # for k in macineList:
        # print(sum(k.getjobCPUUsage()),"===",k.ph.phCPUCap)
        # print("=======")

        for machine in macineList:
            if (len(machine.job)) > 0:
                cpuUsage, MemUsage = calacualteOverAllCapacity(machine)
                if (cpuUsage < threshold and MemUsage < threshold):
                    cpuCapPlusNewJOB, memCapPlusNewJOB = calacualteOverAllCapacityPlusAddedJob(machine, job)
                    PercentClass1CPU, PercentClass1Mem, PercentClass2CPU, PercentClass2Mem, PercentClass3CPU, PercentClass3Mem = calculatePercentageOfClassesForeEachNode(
                        machine, threshold)
                    if cpuCapPlusNewJOB < threshold and memCapPlusNewJOB < threshold:
                        if job.getLabel() == 1 and PercentClass1CPU < thersholForUtiClass1CPU and PercentClass1Mem < thersholForUtiClass1Mem:  # need to add the job based on the type
                            # print('here percentage1',PercentClass1CPU,PercentClass1Mem)
                            machine.addJob(job.getjobID(), job.getCPU(), job.getMemory(), job.getStartTime()
                                           , job.getEndTime(), job.getDuration(), job.getLabel())
                            # macineList.sort(key=lambda x:x.ph.phCPUCap)
                            placeTheJob = placeTheJob + 1
                            schedulStatus.append(True)
                            hostMachine.append(machine)
                            break
                        elif job.getLabel() == 0 and PercentClass2CPU < thersholForUtiClass2CPU and PercentClass2Mem < thersholForUtiClass2Mem:
                            # print('here percentage1',PercentClass1CPU,PercentClass1Mem)
                            machine.addJob(job.getjobID(), job.getCPU(), job.getMemory(), job.getStartTime()
                                           , job.getEndTime(), job.getDuration(), job.getLabel())
                            # macineList.sort(key=lambda x:x.ph.phCPUCap)
                            placeTheJob = placeTheJob + 1
                            schedulStatus.append(True)
                            hostMachine.append(machine)
                            break
                        elif job.getLabel() == 3 and PercentClass3CPU < thersholForUtiClass3CPU and PercentClass3Mem < thersholForUtiClass3Mem:
                            # print('here percentage1',PercentClass1CPU,PercentClass1Mem)
                            machine.addJob(job.getjobID(), job.getCPU(), job.getMemory(), job.getStartTime()
                                           , job.getEndTime(), job.getDuration(), job.getLabel())
                            # macineList.sort(key=lambda x:x.ph.phCPUCap)
                            placeTheJob = placeTheJob + 1
                            schedulStatus.append(True)
                            hostMachine.append(machine)
                            break
        if placeTheJob == 0:
            for machine in macineList:
                cpuUsage1, MemUsage1 = calacualteOverAllCapacity(machine)
                if (cpuUsage1 < threshold and MemUsage1 < threshold):
                    cpuCapPlusNewJOB1, memCapPlusNewJOB1 = calacualteOverAllCapacityPlusAddedJob(machine, job)
                    PercentClass1CPU, PercentClass1Mem, PercentClass2CPU, PercentClass2Mem, PercentClass3CPU, PercentClass3Mem = calculatePercentageOfClassesForeEachNode(
                        machine, threshold)
                    if cpuCapPlusNewJOB1 < threshold and memCapPlusNewJOB1 < threshold:
                        if job.getLabel() == 1 and PercentClass1CPU < .55 and PercentClass1Mem < 0.55:  # need to add the job based on the type
                            # print('here percentage1',PercentClass1CPU,PercentClass1Mem)
                            machine.addJob(job.getjobID(), job.getCPU(), job.getMemory(), job.getStartTime()
                                           , job.getEndTime(), job.getDuration(), job.getLabel())
                            # macineList.sort(key=lambda x:x.ph.phCPUCap)
                            placeTheJob = placeTheJob + 1
                            schedulStatus.append(True)
                            hostMachine.append(machine)
                            break
                        elif job.getLabel() == 0 and PercentClass2CPU < .25 and PercentClass2Mem < 0.25:
                            # print('here percentage1',PercentClass1CPU,PercentClass1Mem)
                            machine.addJob(job.getjobID(), job.getCPU(), job.getMemory(), job.getStartTime()
                                           , job.getEndTime(), job.getDuration(), job.getLabel())
                            # macineList.sort(key=lambda x:x.ph.phCPUCap)
                            placeTheJob = placeTheJob + 1
                            schedulStatus.append(True)
                            hostMachine.append(machine)
                            break
                        elif job.getLabel() == 3 and PercentClass3CPU < .50 and PercentClass3Mem < 0.50:
                            # print('here percentage1',PercentClass1CPU,PercentClass1Mem)
                            machine.addJob(job.getjobID(), job.getCPU(), job.getMemory(), job.getStartTime()
                                           , job.getEndTime(), job.getDuration(), job.getLabel())
                            # macineList.sort(key=lambda x:x.ph.phCPUCap)
                            placeTheJob = placeTheJob + 1
                            schedulStatus.append(True)
                            hostMachine.append(machine)
                            break
    return schedulStatus, hostMachine


def MigratedFromOverloadedToUnderloaded(schedul, ListOfOverloadedPMSAdresses, threshold):
    SLAV = []
    schedulStatus = []
    hostMachine = []
    migratedJobsData = []
    scheduledJobFromOverloadedToUnderloaded = False
    # sorting decreasing
    for i in ListOfOverloadedPMSAdresses:
        j = schedul[i]
        # j.job.sort(key=lambda x: x.jobCPUUsage)
        j.job.sort(key=lambda x: (x.priorty, x.jobCPUUsage))
        print(len(j.job))
        # print('Utilization Percent',(sum(calculateCPUUtilization1([j])))/len(calculateCPUUtilization1([j])))
        for k in j.job:
            if (sum(calculateCPUUtilization1([j]))) / len(calculateCPUUtilization1([j])) <= threshold:
                # print('break1')
                break
            else:
                schedulStatus, hostMachine = schedulingIntLeastLoaded(schedul, [k], threshold)
                if schedulStatus:
                    # print('done moving=',k.jobCPUUsage)
                    SLAV.append((sum(j.getjobCPUUsage()) * .1) / k.getCPU())
                    # print('Here',(sum(j.getjobCPUUsage())*.1),k.getCPU(),(sum(j.getjobCPUUsage())*.1)/k.getCPU())
                    # call function that requrd the data of migration process
                    # migratedJobsData.append(migratedVMsData(k, j, hostMachine))
                    migratedJobsData.append((k.getCPU(), k.getMemory()))
                    j.deletJob(0)


                else:
                    print("fild")

    return SLAV, migratedJobsData


# the algorithm for palcing the arrival jobs into the least loaded PM
# return the adderss of the loaded machines
def findTheLoadedPM(machines):
    addresOflodedPM = []
    for i, j in (zip(machines, (range(len(machines))))):
        if len(i.job) > 0:
            addresOflodedPM.append(j)

    return addresOflodedPM


# =============================================
# v.1 of our algorithm
def schedulingIntLeastLoaded(macineList, jobList, threshold):
    for job in jobList:
        placeTheJob = 0
        schedulStatus = []
        hostMachine = []
        macineList.sort(key=lambda x: (x.getPhCPUCap() - (sum(x.getjobCPUUsage()))), reverse=True)
        # for k in macineList:
        # print(sum(k.getjobCPUUsage()),"===",k.ph.phCPUCap)
        # print("=======")

        for machine in macineList:
            if (len(machine.job)) > 0:
                cpuUsage, MemUsage = calacualteOverAllCapacity(machine)
                if (cpuUsage < threshold and MemUsage < threshold):
                    cpuCapPlusNewJOB, memCapPlusNewJOB = calacualteOverAllCapacityPlusAddedJob(machine, job)
                    if (
                            cpuCapPlusNewJOB < threshold and memCapPlusNewJOB < threshold):  # need to add the job based on the type
                        # print('here loaded',cpuCapPlusNewJOB)
                        machine.addJob(job.getjobID(), job.getCPU(), job.getMemory(), job.getStartTime()
                                       , job.getEndTime(), job.getDuration(), job.getLabel(),job.getPriority())
                        # macineList.sort(key=lambda x:x.ph.phCPUCap)
                        placeTheJob = placeTheJob + 1
                        schedulStatus.append(True)
                        hostMachine.append(machine)
                        break
        if placeTheJob == 0:
            for machine in macineList:
                cpuUsage1, MemUsage1 = calacualteOverAllCapacity(machine)
                if (cpuUsage1 < threshold and MemUsage1 < threshold):
                    cpuCapPlusNewJOB1, memCapPlusNewJOB1 = calacualteOverAllCapacityPlusAddedJob(machine, job)
                    if (cpuCapPlusNewJOB1 < threshold and memCapPlusNewJOB1 < threshold):
                        # print('here unloaded',cpuCapPlusNewJOB1)
                        machine.addJob(job.getjobID(), job.getCPU(), job.getMemory(), job.getStartTime()
                                       , job.getEndTime(), job.getDuration(), job.getLabel(),job.getPriority())
                        # macineList.sort(key=lambda x:x.ph.phCPUCap)
                        placeTheJob = placeTheJob + 1
                        schedulStatus.append(True)
                        hostMachine.append(machine)
                        break
    return schedulStatus, hostMachine
def migratedVMsData(job, hostMachine, destnationMachine):
    # print("number of destination Machine",len(destnationMachine))

    return MigratedJobs(job.getjobID(), job.getMemory(), job.getMemory(),hostMachine.getMachineID(),
                        destnationMachine[0].getMachineID(), hostMachine.getPodNumber(),
                        destnationMachine[0].getPodNumber(), hostMachine.getSwitchNumber(),
                        destnationMachine[0].getSwitchNumber())