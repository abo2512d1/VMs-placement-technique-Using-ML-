#job class
class Job:
    def __init__(self,jobid,jopCPU,jobMem,jobStTime,jobEnTime,jobDur,label,priorty):
        self.jobID=jobid
        self.jobCPUUsage=jopCPU
        self.jobMemUsage=jobMem
        self.jobStartTime=jobStTime
        self.jobEndTime=jobEnTime
        self.jobDuration=jobDur
        self.label=label
        self.priorty=priorty
    def printJob(self):
        print(self.jobID,self.jobCPUUsage,self.jobMemUsage,self.jobStartTime,self.jobEndTime,self.jobDuration,self.label)
    def getjobID(self):
        return self.jobID
    def getCPU(self):
        return self.jobCPUUsage
    def getMemory(self):
        return self.jobMemUsage
    def getStartTime(self):
        return self.jobStartTime
    def getEndTime(self):
        return self.jobEndTime
    def getDuration(self):
        return self.jobDuration
    def getLabel(self):
        return self.label
    def getPriority(self):
        return self.priorty