class Phmachine:

    def __init__(self, ID, CPUCap, MemCap):
        self.phID = ID
        self.phCPUCap = CPUCap
        self.phMemCap = MemCap
        self.podNumber = -1
        self.switchNumber = -1
        self.portNumber = -1

    def setPodNumber(self, podNum):
        self.podNumber = podNum

    def setSwitchNumber(self, switchNum):
        self.switchNumber = switchNum

    def setPortNumber(self, portNum):
        self.portNumber = portNum

    def show(self):
        print('machineID=', self.phID,
              '\nmachine CPU capacity', self.phCPUCap,
              '\nmachine CPU capacity', self.phMemCap)

    def getMachineID(self):
        return self.phID