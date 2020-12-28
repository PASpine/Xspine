from __future__ import print_function
import torch.optim as optim
from torch.utils.data import DataLoader
import dataset
from model import *
from yolo_loss import *
from config import Config
from utils import *
import torch.nn as nn

#Parameters
use_cuda = torch.cuda.is_available()
eps = 1e-9

def adjust_learning_rate(optimizer, batch, steps, scales, lr):
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / batch_size
    return lr

def train_epoch(epoch, train_loader, yolo_config, device, writer=None):
    global processed_batches
    t0 = time.time()
    if yolo_config.solver == 'sgd':
        logging('SGD epoch %d, processed %d samples, lr %f' % (epoch, epoch * len(train_loader.dataset), yolo_config.learning_rate))
    elif yolo_config.solver == 'adam':
        logging('Adam epoch %d, processed %d samples, lr %f' % (epoch, epoch * len(train_loader.dataset), yolo_config.learning_rate))

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        processed_batches = processed_batches + 1

        data = data.cuda()
        optimizer.zero_grad()

        output = model(data)
        # print(output[0].shape, output[1].shape, output[2].shape, target.shape)
        Outputs = [output[0], output[1], output[2]]

        loss, nGT, nCorrect, nProposals = YoloLoss(Outputs, target, yolo_config.num_classes,
                                                   processed_batches * batch_size, device)


        total_loss = loss[0] + loss[1] + loss[2]

        if yolo_config.tensorboard:
            writer.add_scalars('PR_1', {'recall': float(nCorrect[0] / (nGT + eps)), 'precision': float(nCorrect[0] / (nProposals[0] + eps))}, processed_batches)
            writer.add_scalars('PR_2', {'recall': float(nCorrect[1] / (nGT + eps)), 'precision': float(nCorrect[1] / (nProposals[1] + eps))}, processed_batches)
            writer.add_scalars('PR_3', {'recall': float(nCorrect[2] / (nGT + eps)), 'precision': float(nCorrect[2] / (nProposals[2] + eps))}, processed_batches)
            writer.add_scalar('tra_loss_1', loss[0].item(), processed_batches)
            writer.add_scalar('tra_loss_2', loss[1].item(), processed_batches)
            writer.add_scalar('tra_loss_3', loss[2].item(), processed_batches)
            writer.add_scalar('total_tra_loss', total_loss.item(), processed_batches)

        total_loss.backward()
        optimizer.step()

    print('')
    t1 = time.time()
    logging('training with %f samples/s' % (len(train_loader.dataset) / (t1 - t0)))
    if (epoch + 1) % yolo_config.save_interval == 0:
        logging('save weights to %s/%06d.pkl' % (yolo_config.backupDir, epoch + 1))
        save_model_filename = '%s/%06d.pkl' % (yolo_config.backupDir, epoch + 1)
        model.module.seen = (epoch + 1) * len(train_loader.dataset)
        model.module.save_weights(save_model_filename)

if __name__ == '__main__':
    # Training settings
    yolo_config = Config()
    batch_size = yolo_config.batch_size
    if not os.path.exists(yolo_config.backupDir):
        os.mkdir(yolo_config.backupDir)
    kwargs = {'num_workers': yolo_config.num_workers, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda:%s" % str(yolo_config.gpus[0]) if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(yolo_config.gpus[0])
        print("GPU is available!")

    model = YOLO(yolo_config.num_classes, yolo_config.in_channels)
    model.width = yolo_config.init_width
    model.height = yolo_config.init_height
    model = nn.DataParallel(model, device_ids=yolo_config.gpus)
    model = model.cuda()

    # yolo_loss = YoloLoss(yolo_config.num_classes, iou_loss_thresh=0.5, anchors=yolo_config.get_anchors()).cuda()

    writer = None
    if yolo_config.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(yolo_config.logsDir)

    if yolo_config.solver == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=yolo_config.learning_rate/batch_size, momentum=yolo_config.momentum, weight_decay=yolo_config.decay*batch_size)
    elif yolo_config.solver == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=yolo_config.learning_rate, betas=yolo_config.betas, weight_decay=yolo_config.decay, amsgrad=True)
    else:
        print('No %s solver! Please check your config file!' % (yolo_config.solver))
        exit()
    nsamples = os.listdir(yolo_config.imgDirPath).__len__()
    ################################
    #         load weights         #
    ################################
    if yolo_config.weightFile != 'none':
        # model.module.load_weights(yolo_config.weightFile)
        model.module.load_pretrained_weights(yolo_config.weightFile)
    else:
        model.module.seen = 0
    processed_batches = model.module.seen / batch_size
    init_epoch = int(model.module.seen / nsamples)

    train_loader = DataLoader(
        dataset.listTxTDataset(imgdirpath=yolo_config.imgDirPath,
                            shape=(model.module.width, model.module.height),
                            transform=transforms.Compose([
                                # transforms.Resize((512, 512)),
                                transforms.ToTensor(),
                            ])
                            ),
        batch_size=yolo_config.batch_size, shuffle=True, **kwargs)
    for epoch in range(init_epoch, yolo_config.max_epochs):
        train_epoch(epoch, train_loader, yolo_config, device, writer)

    if yolo_config.tensorboard:
        writer.close()

    print('Done!')