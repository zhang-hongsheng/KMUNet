import argparse
import shutil
import sys
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from metrics_2 import calculate_metrics
from loader import *
from losses import DiceLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim import Adam
from model.KMUNet import KMUNet


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                     default=r'', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='KMUNet', help='model_name')
parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
parser.add_argument("--opts", help="Modify config options. ", default=None, nargs='+')
parser.add_argument('--max_epochs', type=int,
                    default=400, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--labeled_num', type=int, default=54381,
                    help='labeled data')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

def train(args):

    current_state_path = os.path.join(snapshot_path, 'current_state.pth.tar')
    best_state_path = os.path.join(snapshot_path, 'best_state.pth.tar')

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)

    print('#----------Prepareing Models----------#')

    model = KMUNet(depths=[1, 2, 6, 2], hidden_dims=[64, 128, 256, 512]).to(device)

    my_transform = My_Train_Transforms()
    db_train = BaseDataSets_ISIC(base_dir=args.root_path, split="train", transform=my_transform, num=None)
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True,
                             num_workers=32, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=args.batch_size, shuffle=False, num_workers=32, pin_memory=True)

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr,
    #                       momentum=0.9, weight_decay=0.0001)

    # optimizer = AdamW(model.parameters(),
    #                   lr=0.001,
    #                   betas=(0.9, 0.999),
    #                   eps=1e-8,
    #                   weight_decay=0.01,
    #                   amsgrad=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs)

    writer = SummaryWriter(snapshot_path + '/log')
    start_epoch = 0
    iter_num = 0
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0

    if os.path.exists(current_state_path):
        print("Loading current state: '{}'".format(current_state_path))
        current_state = torch.load(current_state_path)
        model.load_state_dict(current_state['model_parameters'])
        optimizer.load_state_dict(current_state['optimizer'])
        iter_num = current_state['item']
        start_epoch = current_state['epoch']
        best_performance = current_state['best_iou']

    for epoch in range(start_epoch, args.max_epochs):
        torch.cuda.empty_cache()
        model.train()

        logging.info('Epoch{} train start'.format(epoch + 1))
        train_tbar = tqdm(trainloader, ncols=160, dynamic_ncols=False)

        train_loss_sum = AverageMeter()
        train_loss_ce = AverageMeter()
        train_loss_dice = AverageMeter()

        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'].float(), sampled_batch['label'].float()
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
            outputs = model(volume_batch)

            outputs_soft = torch.softmax(outputs, dim=1)

            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            loss = 0.5 * (loss_dice + loss_ce)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            train_loss_sum.update(loss.item())
            train_loss_ce.update(loss_ce.item())
            train_loss_dice.update(loss_dice.item())

            train_tbar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {iter:4}/{iters:4}. Loss_sum:{loss:.4f}. Loss_ce:{ce:.4f}. Loss_unsup:{dice:.4f}".format(
                    epoch=epoch + 1, epochs=args.max_epochs, iter=i_batch + 1, iters=len(trainloader),
                    loss=train_loss_sum.avg, ce=train_loss_ce.avg, dice=train_loss_dice.avg))
            train_tbar.update(1)
        train_tbar.close()

        scheduler.step()

        logging.info("Epoch{} train finish".format(epoch + 1))
        logging.info('Epoch{} validation start'.format(epoch + 1))

        model.eval()

        val_tbar = tqdm(valloader, ncols=160, dynamic_ncols=False, desc='Val: Epoch {}/{}'.format({epoch + 1}, {args.max_epochs}),
                        postfix=None, mininterval=0.3)

        val_loss_total_sum = AverageMeter()
        val_loss_ce_sum = AverageMeter()
        val_loss_dice_sum = AverageMeter()

        all_iou = []
        all_dice = []
        all_recall = []
        all_precision = []
        all_specificity = []
        all_accuracy = []

        with torch.no_grad():
            for val_i_batch, sampled_batch in enumerate(valloader):

                image_batch, label_batch = sampled_batch['image'].float(), sampled_batch['label'].float()
                image_batch, label_batch = image_batch.to(device), label_batch.to(device)

                outputs = model(image_batch)
                outputs_soft = torch.softmax(outputs, dim=1)

                val_loss_ce = ce_loss(outputs, label_batch[:].long())
                val_loss_dice = dice_loss(outputs, label_batch.unsqueeze(1), softmax=True)
                val_total_loss = 0.5 * (val_loss_ce + val_loss_dice)

                val_loss_total_sum.update(val_total_loss.item())
                val_loss_ce_sum.update(val_loss_ce.item())
                val_loss_dice_sum.update(val_loss_dice.item())

                _, predicted = torch.max(outputs_soft, 1)

                results = calculate_metrics(predicted, label_batch, num_classes=args.num_classes)

                all_iou.extend(results['iou'])
                all_dice.extend(results['dice'])
                all_recall.extend(results['recall'])
                all_precision.extend(results['precision'])
                all_specificity.extend(results['specificity'])
                all_accuracy.extend(results['accuracy'])

                val_tbar.set_description(
                    "Val Epoch: {epoch}/{epochs:4}. Iter: {iter:4}/{iters:4}. Loss_sum:{loss:.4f}. Loss_ce:{ce:.4f}. Loss_dice:{dice:.4f}".format(
                        epoch=epoch + 1, epochs=args.max_epochs, iter=val_i_batch + 1, iters=len(valloader),
                        loss=val_loss_total_sum.avg, ce=val_loss_ce_sum.avg, dice=val_loss_dice_sum.avg))
                val_tbar.update(1)
            val_tbar.close()

            avg_iou = sum(all_iou) / len(all_iou)
            avg_dice = sum(all_dice) / len(all_dice)
            avg_recall = sum(all_recall) / len(all_recall)
            avg_precision = sum(all_precision) / len(all_precision)
            avg_specificity = sum(all_specificity) / len(all_specificity)
            avg_accuracy = sum(all_accuracy) / len(all_accuracy)

        writer.add_scalar('info/val_mean_iou', avg_iou, epoch + 1)
        writer.add_scalar('info/val_mean_dice', avg_dice, epoch + 1)
        writer.add_scalar('info/val_recall', avg_recall, epoch + 1)
        writer.add_scalar('info/val_precision', avg_precision, epoch + 1)
        writer.add_scalar('info/val_specificity', avg_specificity, epoch + 1)
        writer.add_scalar('info/val_accuracy', avg_accuracy, epoch + 1)

        logging.info("Epoch{} validation finish".format(epoch + 1))
        logging.info("Val iou:{:.4f}. Val dice:{:.4f}. Val recall:{:.4f}. Val precision:{:.4f}. Val spec:{:.4f}. "
                     "val acc:{:.4f}".format(avg_iou, avg_dice, avg_recall, avg_precision, avg_specificity,
                                             avg_accuracy))
        logging.info('Save model checkpoint')
        model.train()

        if avg_iou > best_performance:
            best_performance = avg_iou
            best_state = {'model_parameters': model.state_dict()}
            torch.save(best_state, best_state_path)
            logging.info('Best state save successfully!')
        current_state = {
            'model_parameters': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'item': len(trainloader) * (epoch + 1),
            'iou': avg_iou,
            'best_iou': best_performance,
        }
        torch.save(current_state, current_state_path)
        logging.info('Current state save successfully!')

    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../{}/{}".format(
        args.model, args.exp)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    name = os.path.basename(__file__)
    shutil.copyfile(name, os.path.join(snapshot_path, name))
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args)
