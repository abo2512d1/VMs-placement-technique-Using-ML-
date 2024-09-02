from phmachine import *
from job import *
class Schedual:

    def __init__(self, iid, cpu, mem):
        self.ph = Phmachine(iid, cpu, mem)
        self.job = []

    def addJob(self, ID, cpu, mem, stime, etime, dur, lebl,priorty):
        self.job.append(Job(ID, cpu, mem, stime, etime, dur, lebl,priorty))

    def showMachineCapacity(self):
        print('Machine ID', self.ph.phID,
              '\nMachine CPU cap', self.ph.phCPUCap,
              '\nMachine Memory Cap', self.ph.phMemCap)

    def showjobRequ(self):
        for i in self.job:
            print(
                i.jobCPUUsage,
                i.jobMemUsage,
                i.jobStartTime,
                i.jobEndTime,
                i.jobDuration,
                i.label)

    def deletJob(self, i):
        del self.job[i]

    def getMachineID(self):
        return self.ph.phID

    def clearr(self):
        self.job.clear()

    def getPhCPUCap(self):
        return self.ph.phCPUCap

    def getPhMemory(self):
        return self.ph.phMemCap

    def getLabel(self):
        return self.ph.label

    def getPodNumber(self):
        return self.ph.podNumber

    def getSwitchNumber(self):
        return self.ph.switchNumber

    def getPortNumber(self):
        return self.ph.portNumber

    def getjobCPUUsage(self):
        cpuUsage = []
        for i in range(len(self.job)):
            cpuUsage.append(self.job[i].jobCPUUsage)
        return cpuUsage

    def getjobMemUsage(self):
        MemUsage = []
        for i in range(len(self.job)):
            MemUsage.append(self.job[i].jobMemUsage)
        return MemUsage

    def getjobStartTime(self):
        StartTime = []
        for i in range(len(self.job)):
            StartTime.append(self.job[i].jobStartTime)
        return StartTime

    def getjobEndTime(self):
        EndTime = []
        for i in range(len(self.job)):
            EndTime.append(self.job[i].jobEndTime)
        return EndTime

    def getjobDuration(self):
        Duration = []
        for i in range(len(self.job)):
            Duration.append(self.job[i].jobDuration)
        return Duration