class MigratedJobs:
    def __init__(self, jobid, jobMem, jobAsMemory, HServer, DServer, HSPod, DSPod, HSSwitch, DSSwitch):
        self.jobID = jobid
        self.jobMemUsage = jobMem
        self.jobAssignMemory = jobAsMemory
        self.HostServer = HServer
        self.DestinationServer = DServer
        self.HostServerPod = HSPod
        self.DestnationServerPod = DSPod
        self.HostServerSwitch = HSSwitch
        self.DestnationServerSwitch = DSSwitch

    def printMigratedVMsData(self):
        print("jobID", self.jobID, "jobMem", self.jobMemUsage, "Assign Memory", self.jobAssignMemory, "HServer",
              self.HostServer, "DServer", self.DestinationServer,
              "HSPod", self.HostServerPod, "DSPod", self.DestnationServerPod)

    def printMigratedVMsDataWithoutLable(self):
        print(self.jobID, self.jobMemUsage, self.HostServer, self.DestinationServer,
              self.HostServerPod, self.DestnationServerPod, self.HostServerSwitch, DestnationServerSwitch)