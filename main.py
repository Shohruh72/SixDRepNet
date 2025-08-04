# https://github.com/Shohruh72
import argparse
import copy
import csv
import os

import cv2
import tqdm
from timm import utils
from torch.utils import data
from face_detection import RetinaFace

from nets import nn
from utils.util import *
from utils.datasets import Datasets


def lr(args):
    return 5E-5 * args.batch_size * args.world_size / 64


def train(args):
    weight = f'./weights/{args.model_name}.pth'
    model = nn.HPE(args.model_name, weight, False, True).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr(args))

    ema = nn.EMA(model) if args.local_rank == 0 else None

    sampler = None
    dataset = Datasets(f'{args.data_dir}', '300W_LP', get_transforms(args, True), True)

    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(dataset, args.batch_size, sampler is None, sampler=sampler, num_workers=8, pin_memory=True)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    best = float('inf')
    num_steps = len(loader)
    criterion = ComputeLoss().cuda()
    amp_scale = torch.cuda.amp.GradScaler()
    scheduler = CosineLR(args, optimizer)
    with open('weights/step.csv', 'w') as log:
        if args.local_rank == 0:
            logger = csv.DictWriter(log, fieldnames=['epoch', 'Loss', 'Pitch', 'Yaw', 'Roll'])
            logger.writeheader()
        for epoch in range(args.epochs):
            model.train()

            if args.distributed:
                sampler.set_epoch(epoch)

            p_bar = loader
            avg_loss = AverageMeter()

            if args.local_rank == 0:
                print(('\n' + '%10s' * 3) % ('epoch', 'memory', 'loss'))
                p_bar = tqdm.tqdm(iterable=p_bar, total=num_steps)

            for samples, labels in p_bar:
                samples = samples.cuda()
                labels = labels.cuda()

                with torch.cuda.amp.autocast():
                    outputs = model(samples)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()

                amp_scale.scale(loss).backward()

                amp_scale.step(optimizer)
                amp_scale.update(None)
                if ema:
                    ema.update(model)

                if args.distributed:
                    loss = utils.reduce_tensor(loss.data, args.world_size)
                avg_loss.update(loss.item(), samples.size(0))
                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'
                    s = ('%10s' * 2 + '%10.3g') % (f'{epoch + 1}/{args.epochs}', memory, avg_loss.avg)
                    p_bar.set_description(s)

            scheduler.step(epoch, optimizer)

            if args.local_rank == 0:
                last = test(args, ema.ema)

                logger.writerow({'Pitch': str(f'{last[0]:.3f}'),
                                 'Yaw': str(f'{last[1]:.3f}'),
                                 'Roll': str(f'{last[2]:.3f}'),
                                 'Loss': str(f'{avg_loss.avg:.3f}'),
                                 'epoch': str(epoch + 1).zfill(3)})
                log.flush()
                is_best = sum(last) < best

                if is_best:
                    best = sum(last)
                save = {'epoch': epoch, 'model': copy.deepcopy(ema.ema).half()}
                torch.save(save, f'weights/last.pt')

                if is_best:
                    torch.save(save, f'weights/best.pt')
                del save

    if args.local_rank == 0:
        strip_optimizer('./weights/best.pt')
        strip_optimizer('./weights/last.pt')
    torch.cuda.empty_cache()


@torch.no_grad()
def test(args, model=None):
    if model is None:
        model = torch.load('./weights/best.pt', 'cuda')
        model = model['model'].float()
    model.eval()

    dataset = Datasets(f'{args.data_dir}', 'AFLW2K', get_transforms(args, False), False)
    loader = data.DataLoader(dataset, batch_size=2)
    total, y_error, p_error, r_error = 0, 0.0, 0.0, 0.0
    for sample, label in tqdm.tqdm(loader, ('%10s' * 3) % ('Pitch', 'Yaw', 'Roll')):
        sample = sample.cuda()
        total += label.size(0)

        p_gt = label[:, 0].float() * 180 / np.pi
        y_gt = label[:, 1].float() * 180 / np.pi
        r_gt = label[:, 2].float() * 180 / np.pi

        output = model(sample)
        euler = compute_euler(output) * 180 / np.pi

        p_pred = euler[:, 0].cpu()
        y_pred = euler[:, 1].cpu()
        r_pred = euler[:, 2].cpu()

        p_error += torch.sum(torch.min(torch.stack((torch.abs(p_gt - p_pred),
                                                    torch.abs(p_pred + 360 - p_gt),
                                                    torch.abs(p_pred - 360 - p_gt),
                                                    torch.abs(p_pred + 180 - p_gt),
                                                    torch.abs(p_pred - 180 - p_gt))), 0)[0])

        y_error += torch.sum(torch.min(torch.stack((torch.abs(y_gt - y_pred),
                                                    torch.abs(y_pred + 360 - y_gt),
                                                    torch.abs(y_pred - 360 - y_gt),
                                                    torch.abs(y_pred + 180 - y_gt),
                                                    torch.abs(y_pred - 180 - y_gt))), 0)[0])

        r_error += torch.sum(torch.min(torch.stack((torch.abs(r_gt - r_pred),
                                                    torch.abs(r_pred + 360 - r_gt),
                                                    torch.abs(r_pred - 360 - r_gt),
                                                    torch.abs(r_pred + 180 - r_gt),
                                                    torch.abs(r_pred - 180 - r_gt))), 0)[0])

    p_error, y_error, r_error = p_error / total, y_error / total, r_error / total
    print(('%10s' * 3) % (f'{p_error:.3f}', f'{y_error:.3f}', f'{r_error:.3f}'))

    model.float()  # for training
    return p_error, y_error, r_error


@torch.no_grad()
def demo(args):
    model = torch.load('./weights/best.pt', map_location='cuda')['model'].float()
    model.half()
    model.eval()
    detector = FaceDetector('./weights/detection.onnx')
    cap = cv2.VideoCapture('vid2.mp4')

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes = detector.detect(frame, (640, 640))
        boxes = boxes.astype('int32')

        for idx, box in enumerate(boxes):
            x_min = box[0]
            y_min = box[1]
            x_max = box[2]
            y_max = box[3]
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            x_min = max(0, x_min - int(0.2 * bbox_height))
            y_min = max(0, y_min - int(0.2 * bbox_width))
            x_max = x_max + int(0.2 * bbox_height)
            y_max = y_max + int(0.2 * bbox_width)

            img = Image.fromarray(frame[y_min:y_max, x_min:x_max]).convert('RGB')
            img = get_transforms(args, False)(img).cuda()
            img = img.unsqueeze(0)
            img = img.cuda().half()

            c = cv2.waitKey(1)
            if c == 27:
                break
            output = model(img)
            output = compute_euler(output) * 180 / np.pi
            p_pred_deg = output[:, 0].cpu()
            y_pred_deg = output[:, 1].cpu()
            r_pred_deg = output[:, 2].cpu()

            plot_pose_cube(frame, y_pred_deg, p_pred_deg, r_pred_deg, x_min + int(.5 * (
                    x_max - x_min)), y_min + int(.5 * (y_max - y_min)), size=bbox_width)
            print(p_pred_deg)
        cv2.imshow("Demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Head Pose Estimation')
    parser.add_argument('--model_name', type=str, default='a2')
    parser.add_argument('--data_dir', type=str, default='../../Datasets/HPE')
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--demo', default=True, action='store_true')
    args = parser.parse_args()

    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1
    setup_seed()
    setup_multi_processes()
    os.makedirs('weights', exist_ok=True)

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.train:
        train(args)
    if args.test:
        test(args)
    if args.demo:
        demo(args)


if __name__ == "__main__":
    main()
