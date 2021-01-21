import datetime
import os
import random
import impath
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
from datasets import wp
from models import *
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d, DiceLoss, make_one_hot
import logging

cudnn.benchmark = False

ckpt_path = './ckpt'
exp_name = 'voc-fcn8s_dice'
print(os.path.join(ckpt_path, 'exp2', exp_name))
writer = SummaryWriter(os.path.join(ckpt_path, 'exp2', exp_name))

args = {
    'epoch_num': 300,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'momentum': 0.95,
    'lr_patience': 5,  # large patience denotes fixed lr
    'snapshot': '',  # empty string denotes learning from scratch
    'print_freq': 20,
    'val_save_to_img_file': False,
    'val_img_sample_rate': 1  # randomly sample some validation results to display
}


def cuda_setup(cuda=False, gpu_idx=0):
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_device(gpu_idx)
    else:
        device = torch.device('cpu')
    return device


def main(train_args):
    backbone = ResNet()
    backbone.load_state_dict(torch.load(
        './weight/resnet34-333f7ec4.pth'), strict=False)
    net = Decoder34(num_classes=13, backbone=backbone).cuda()
    D = Discriminator(input_channels=16).cuda()
    if len(train_args['snapshot']) == 0:
        curr_epoch = 1
        train_args['best_record'] = {
            'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}
    else:
        print('training resumes from ' + train_args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(
            ckpt_path, exp_name, train_args['snapshot'])))
        split_snapshot = train_args['snapshot'].split('_')
        curr_epoch = int(split_snapshot[1]) + 1
        train_args['best_record'] = {'epoch': int(split_snapshot[1]), 'val_loss': float(split_snapshot[3]),
                                     'acc': float(split_snapshot[5]), 'acc_cls': float(split_snapshot[7]),
                                     'mean_iu': float(split_snapshot[9]), 'fwavacc': float(split_snapshot[11])}

    net.train()
    D.train()

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = extended_transforms.MaskToTensor()
    restore_transform = standard_transforms.Compose([
        extended_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage(),
    ])
    visualize = standard_transforms.Compose([
        standard_transforms.Scale(400),
        standard_transforms.CenterCrop(400),
        standard_transforms.ToTensor()
    ])

    train_set = wp.Wp('train', transform=input_transform,
                      target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=8,
                              num_workers=4, shuffle=True)
    # val_set = wp.Wp('val', transform=input_transform,
    #                 target_transform=target_transform)
    # XR：所以这里本来就不能用到val？这里为什么不用一个val的数据集呢？
    val_loader = DataLoader(train_set, batch_size=1,
                            num_workers=4, shuffle=False)
    criterion = DiceLoss().cuda()
    optimizer_AE = optim.Adam([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * train_args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': train_args['lr'], 'weight_decay': train_args['weight_decay']}
    ], betas=(train_args['momentum'], 0.999))
    optimizer_D = optim.Adam([
        {'params': [param for name, param in D.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * train_args['lr']},
        {'params': [param for name, param in D.named_parameters() if name[-4:] != 'bias'],
         'lr': train_args['lr'], 'weight_decay': train_args['weight_decay']}
    ], betas=(train_args['momentum'], 0.999))

    if len(train_args['snapshot']) > 0:
        optimizer_AE.load_state_dict(torch.load(os.path.join(
            ckpt_path, exp_name, 'opt_' + train_args['snapshot'])))
        optimizer_AE.param_groups[0]['lr'] = 2 * train_args['lr']
        optimizer_AE.param_groups[1]['lr'] = train_args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) +
                      '.txt'), 'w').write(str(train_args) + '\n\n')

    scheduler = ReduceLROnPlateau(
        optimizer_AE, 'min', patience=train_args['lr_patience'], min_lr=1e-10, verbose=True)
    for epoch in range(curr_epoch, train_args['epoch_num'] + 1):
        train(train_loader, net, D, criterion, optimizer_AE,
              optimizer_D, epoch, train_args)
        val_loss = validate(val_loader, net, D, criterion, optimizer_AE, optimizer_D,
                            epoch, train_args, restore_transform, visualize)
        scheduler.step(val_loss)


def train(train_loader, net, D, criterion, optimizer_AE, optimizer_D, epoch, train_args):
    train_loss = AverageMeter()
    curr_iter = (epoch - 1) * len(train_loader)
    for i, data in enumerate(train_loader):
        # 训练D
        D.zero_grad()
        inputs, labels = data
        labels = make_one_hot(labels.unsqueeze(1), wp.num_classes)

        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs = net(inputs)
        # ((a1,a2,...), axis=0)
        origin_outputs = np.concatenate([inputs, outputs], axis=1)  # B,16,H,W
        origin_labels = np.concatenate([inputs, labels], axis=1)  # B,16,H,W
        batch_size = inputs.shape[0]

        output_D = D(origin_labels)  # B
        real_label = torch.ones(batch_size).to(device)  # 定义真实的图片label为1
        fake_label = torch.zeros(batch_size).to(device)  # 定义假的图片的label为0
        errD_real = criterion(output_D, real_label)
        errD_real.backward()
        # real_data_score = output_D.mean().item()

        fake_data = vae.decoder(origin_outputs)
        output_D = D(fake_data)  # B
        errD_fake = criterion(output_D, fake_label)
        errD_fake.backward()
        # fake_data_score用来输出查看的，是虚假照片的评分，0最假，1为真
        # fake_data_score = output_D.data.mean()
        errD = errD_real + errD_fake
        optimizer_D.step()
        print('errD', errD.item())

        # 训练AE
        net.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_AE.step()

        train_loss.update(loss.item(), batch_size)
        print('loss', loss.item())
        curr_iter += 1
        writer.add_scalar('train_loss', train_loss.avg, curr_iter)

        if (i + 1) % train_args['print_freq'] == 0:
            print('[epoch %d], [iter %d / %d], [train loss %.5f]' % (
                epoch, i + 1, len(train_loader), train_loss.avg
            ))


def validate(val_loader, net, criterion, optimizer, epoch, train_args, restore, visualize):
    net.eval()

    val_loss = AverageMeter()
    inputs_all, gts_all, predictions_all = [], [], []

    for vi, data in enumerate(val_loader):
        inputs, gts = data
        N = inputs.size(0)
        inputs = inputs.cuda()
        gts_l = make_one_hot(gts.unsqueeze(1), wp.num_classes).cuda()
        outputs = net(inputs)
        predictions = outputs.data.max(1)[1].squeeze_(
            1).squeeze_(0).cpu().numpy()

        val_loss.update(criterion(outputs, gts_l).item(), N)

        if random.random() > train_args['val_img_sample_rate']:
            inputs_all.append(None)
        else:
            inputs_all.append(inputs.squeeze_(0).cpu())
        gts_all.append(gts.squeeze_(0).cpu().numpy())
        predictions_all.append(predictions)

    acc, acc_cls, mean_iu, fwavacc = evaluate(
        predictions_all, gts_all, wp.num_classes)

    if mean_iu > train_args['best_record']['mean_iu']:
        train_args['best_record']['val_loss'] = val_loss.avg
        train_args['best_record']['epoch'] = epoch
        train_args['best_record']['acc'] = acc
        train_args['best_record']['acc_cls'] = acc_cls
        train_args['best_record']['mean_iu'] = mean_iu
        train_args['best_record']['fwavacc'] = fwavacc
        snapshot_name = 'epoch_%d_loss_%.5f_acc_%.5f_acc-cls_%.5f_mean-iu_%.5f_fwavacc_%.5f_lr_%.10f' % (
            epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc, optimizer.param_groups[
                1]['lr']
        )
        torch.save(net.state_dict(), os.path.join(
            ckpt_path, exp_name, snapshot_name + '.pth'))
        torch.save(optimizer.state_dict(), os.path.join(
            ckpt_path, exp_name, 'opt_' + snapshot_name + '.pth'))

        if train_args['val_save_to_img_file']:
            to_save_dir = os.path.join(ckpt_path, exp_name, str(epoch))
            check_mkdir(to_save_dir)

        val_visual = []
        for idx, data in enumerate(zip(inputs_all, gts_all, predictions_all)):
            if data[0] is None:
                continue
            input_pil = restore(data[0])
            gt_pil = wp.colorize_mask(data[1])
            predictions_pil = wp.colorize_mask(data[2])
            if train_args['val_save_to_img_file']:
                input_pil.save(os.path.join(to_save_dir, '%d_input.png' % idx))
                predictions_pil.save(os.path.join(
                    to_save_dir, '%d_prediction.png' % idx))
                gt_pil.save(os.path.join(to_save_dir, '%d_gt.png' % idx))
            val_visual.extend([visualize(input_pil.convert('RGB')), visualize(gt_pil.convert('RGB')),
                               visualize(predictions_pil.convert('RGB'))])
        val_visual = torch.stack(val_visual, 0)
        val_visual = vutils.make_grid(val_visual, nrow=3, padding=5)
        writer.add_image(snapshot_name, val_visual)

    print('--------------------------------------------------------------------')
    print('[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f]' % (
        epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc))

    print('best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [epoch %d]' % (
        train_args['best_record']['val_loss'], train_args['best_record']['acc'], train_args['best_record']['acc_cls'],
        train_args['best_record']['mean_iu'], train_args['best_record']['fwavacc'], train_args['best_record']['epoch']))

    print('--------------------------------------------------------------------')

    writer.add_scalar('val_loss', val_loss.avg, epoch)
    writer.add_scalar('acc', acc, epoch)
    writer.add_scalar('acc_cls', acc_cls, epoch)
    writer.add_scalar('mean_iu', mean_iu, epoch)
    writer.add_scalar('fwavacc', fwavacc, epoch)
    writer.add_scalar('lr', optimizer.param_groups[1]['lr'], epoch)

    net.train()
    return val_loss.avg


if __name__ == '__main__':
    main(args)
    # backbone = ResNet()
    # backbone.load_state_dict(torch.load('../../ckpt/voc-fcn8s/epoch_53_loss_0.67099_acc_0.97743_acc-cls_0.65331_mean-iu_0.45789_fwavacc_0.97293_lr_0.0001000000.pth'), strict=False)
    # net = Decoder34(num_classes=13,backbone=backbone).cuda()
    # net.eval()
    # mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # # train_joint_transform = joint_transforms.Compose([
    # #     joint_transforms.Scale(512),
    # #     joint_transforms.RandomRotate(10),
    # #     joint_transforms.RandomHorizontallyFlip()
    # # ])
    # # sliding_crop = joint_transforms.SlidingCrop(224, 2 / 3., -1)
    # input_transform = standard_transforms.Compose([
    #     standard_transforms.ToTensor(),
    #     standard_transforms.Normalize(*mean_std)
    # ])
    # target_transform = extended_transforms.MaskToTensor()
    # restore_transform = standard_transforms.Compose([
    #     extended_transforms.DeNormalize(*mean_std),
    #     standard_transforms.ToPILImage(),
    # ])
    # visualize = standard_transforms.Compose([
    #     standard_transforms.Scale(400),
    #     standard_transforms.CenterCrop(400),
    #     standard_transforms.ToTensor()
    # ])
    #
    # val_set = wp.Wp('val', transform=input_transform, target_transform=target_transform)
    # val_loader = DataLoader(val_set, batch_size=1, num_workers=4, shuffle=False)
    # inputs_all, gts_all, predictions_all = [], [], []
    #
    # for vi, data in enumerate(val_loader):
    #     inputs, gts = data
    #     N = inputs.size(0)
    #     inputs = Variable(inputs, volatile=True).cuda()
    #     gts = Variable(gts, volatile=True)
    #     gts_l = make_one_hot(gts.unsqueeze(1),wp.num_classes).cuda()
    #     outputs = net(inputs)
    #     predictions = outputs.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
    #
    #
    #     if random.random() > 1:
    #         inputs_all.append(None)
    #     else:
    #         inputs_all.append(inputs.data.squeeze_(0).cpu())
    #     gts_all.append(gts.data.squeeze_(0).cpu().numpy())
    #     predictions_all.append(predictions)
    #
    # acc, acc_cls, mean_iu, fwavacc = evaluate(predictions_all, gts_all, wp.num_classes)
    # print(mean_iu)
