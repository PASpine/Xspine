class Config(object):
    imgDirPath = '/data/rtao/Xspine/data'
    labelDirPath = '/data/rtao/Xspine/data'

    # Weight path or none
    weightFile = "none"
    # Loss visualization
    # ON or OFF
    tensorboard = False
    logsDir = "runs"

    # Train params

    # model save path
    backupDir = "backup"
    max_epochs = 6000
    save_interval = 10
    # e.g. 0,1,2,3
    gpus = [0]
    # multithreading
    num_workers = 2
    batch_size = 1

    # Solver params
    # adma or sgd
    solver = "adam"
    steps = [8000, 16000]
    scales = [0.1, 0.1]
    learning_rate = 3e-4#1e-5
    momentum = 0.9
    decay = 0 #5e-4
    betas = (0.9, 0.98)

    # YoloNet params

    num_classes = 1
    in_channels = 1
    init_width = 448
    init_height = 800

    # anchors1 = [77, 87, 120, 64, 91, 164]
    # anchors2 = [66, 57, 59, 81, 44, 142]
    # anchors3 = [22, 35, 44, 48, 45, 68]
    anchors1 = [88, 98, 93, 110, 99, 124]
    anchors2 = [63, 90, 76, 90, 78, 109]
    anchors3 = [33, 42, 44, 54, 63, 72]
    def get_anchors(self):
        anchor1 = []
        anchor2 = []
        anchor3 = []
        for i in range(len(self.anchors1)):
            anchor1.append(self.anchors1[i] / 32)
        for i in range(len(self.anchors2)):
            anchor2.append(self.anchors2[i] / 16)
        for i in range(len(self.anchors3)):
            anchor3.append(self.anchors3[i] / 8)
        anchors = [anchor1, anchor2, anchor3]
        return anchors