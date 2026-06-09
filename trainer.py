import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from scipy.ndimage import zoom
from utils import DiceLoss
from torchvision import transforms


# ============================================================
# 基于混淆矩阵的分割指标:mIoU / Pixel Accuracy / Recall / Precision
# ============================================================
class SegmentationMetric(object):
    def __init__(self, num_classes, ignore_index=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.float64)

    def update(self, pred, label):
        pred = np.asarray(pred).flatten()
        label = np.asarray(label).flatten()
        mask = (label >= 0) & (label < self.num_classes)
        if self.ignore_index is not None:
            mask &= (label != self.ignore_index)
        idx = self.num_classes * label[mask].astype(np.int64) + pred[mask].astype(np.int64)
        binc = np.bincount(idx, minlength=self.num_classes ** 2)
        self.confusion_matrix += binc.reshape(self.num_classes, self.num_classes)

    def pixel_accuracy(self):
        return np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + 1e-10)

    def class_recall(self):
        # TP / (TP + FN)
        return np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + 1e-10)

    def class_precision(self):
        # TP / (TP + FP)
        return np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=0) + 1e-10)

    def iou_per_class(self):
        inter = np.diag(self.confusion_matrix)
        union = (self.confusion_matrix.sum(axis=1)
                 + self.confusion_matrix.sum(axis=0) - inter)
        return inter / (union + 1e-10)

    def mean_iou(self):
        return float(np.nanmean(self.iou_per_class()))

    def summary(self):
        iou = self.iou_per_class()
        recall = self.class_recall()
        precision = self.class_precision()
        return {
            "pixel_acc": float(self.pixel_accuracy()),
            "mIoU": float(np.nanmean(iou)),
            "iou_per_class": iou,
            "recall_per_class": recall,
            "mean_recall": float(np.nanmean(recall)),
            "precision_per_class": precision,
            "mean_precision": float(np.nanmean(precision)),
        }

    def reset(self):
        self.confusion_matrix.fill(0)


# ============================================================
# 逐体积推理,返回与 label 同形状的预测,用于更新混淆矩阵
# ============================================================
def predict_volume_for_metrics(image, model, patch_size):
    """
    image: torch.Tensor, shape (1, D, H, W) or (1, H, W) from test_vol loader
    返回 numpy 数组,与去掉 batch 维后的 image 空间形状一致
    """
    img = image.squeeze(0).cpu().detach().numpy()

    if img.ndim == 3:  # 3D 体 (D, H, W)
        D, H, W = img.shape
        pred_vol = np.zeros((D, H, W), dtype=np.int64)
        for d in range(D):
            slc = img[d]
            if H != patch_size[0] or W != patch_size[1]:
                slc_in = zoom(slc, (patch_size[0] / H, patch_size[1] / W), order=3)
            else:
                slc_in = slc
            inp = torch.from_numpy(slc_in).unsqueeze(0).unsqueeze(0).float().cuda()
            with torch.no_grad():
                out = model(inp)
                out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0).cpu().numpy()
            if H != patch_size[0] or W != patch_size[1]:
                pred = zoom(out, (H / patch_size[0], W / patch_size[1]), order=0)
            else:
                pred = out
            pred_vol[d] = pred
        return pred_vol

    else:  # 2D (H, W)
        H, W = img.shape
        if H != patch_size[0] or W != patch_size[1]:
            img_in = zoom(img, (patch_size[0] / H, patch_size[1] / W), order=3)
        else:
            img_in = img
        inp = torch.from_numpy(img_in).unsqueeze(0).unsqueeze(0).float().cuda()
        with torch.no_grad():
            out = model(inp)
            out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0).cpu().numpy()
        if H != patch_size[0] or W != patch_size[1]:
            pred = zoom(out, (H / patch_size[0], W / patch_size[1]), order=0)
        else:
            pred = out
        return pred


# ============================================================
# 新增:统计模型参数量、FLOPs、GPU 显存、推理延迟和吞吐量
# ============================================================
def print_model_stats(model, img_size, in_channels=1, device='cuda'):
    """
    打印:
        Model Parameters: xx M, FLOPs: xx GMac
        🚀 Max GPU memory allocated: xx MB
        🚀 Avg latency: xx ms, Throughput: xx images/s
    """
    # ---------- 参数量 ----------
    n_params = sum(p.numel() for p in model.parameters())
    params_str = f"{n_params / 1e6:.2f} M"

    # ---------- FLOPs (优先 ptflops, 失败回退 thop) ----------
    flops_str = "N/A"
    try:
        from ptflops import get_model_complexity_info
        was_training = model.training
        model.eval()
        with torch.cuda.device(0):
            macs, _ = get_model_complexity_info(
                model, (in_channels, img_size, img_size),
                as_strings=True, print_per_layer_stat=False, verbose=False
            )
        flops_str = macs
        if was_training:
            model.train()
    except Exception as e1:
        try:
            from thop import profile
            was_training = model.training
            model.eval()
            dummy = torch.randn(1, in_channels, img_size, img_size).to(device)
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
            flops_str = f"{macs / 1e9:.2f} GMac"
            if was_training:
                model.train()
        except Exception as e2:
            flops_str = f"N/A (ptflops/thop 未安装: {e1} | {e2})"

    print(f"Model Parameters: {params_str}, FLOPs: {flops_str}")

    # ---------- GPU 显存 ----------
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        was_training = model.training
        model.eval()
        with torch.no_grad():
            dummy = torch.randn(1, in_channels, img_size, img_size).to(device)
            _ = model(dummy)
        max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"🚀 Max GPU memory allocated: {max_mem:.2f} MB")

        # ---------- 推理延迟 / 吞吐量 ----------
        repeat = 50
        # warm-up
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy)
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(repeat):
                _ = model(dummy)
        torch.cuda.synchronize()
        end = time.time()
        avg_latency = (end - start) / repeat * 1000  # ms
        throughput = repeat / (end - start)
        print(f"🚀 Avg latency: {avg_latency:.2f} ms, Throughput: {throughput:.2f} images/s")
        if was_training:
            model.train()


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    from utils import test_single_volume
    import logging
    import os
    from torch.utils.data import DataLoader
    from torchvision import transforms

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    # ==================== 训练集 ====================
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print(f"The length of train set is: {len(db_train)}")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    # ==================== 验证集 ====================
    db_val = Synapse_dataset(base_dir='./datasets/Synapse/test_vol_h5',
                             list_dir=args.list_dir, split="test_vol")
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    print(f"The length of val set is: {len(db_val)}")

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)

    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    # ==================== 新增:打印模型统计信息 ====================
    print("\n" + "=" * 60)
    print("📊 Model & Runtime Statistics")
    print("=" * 60)
    # TransUNet 默认输入是 1 通道(灰度切片),如果你模型是 3 通道,请把 in_channels 改成 3
    try:
        # 兼容 DataParallel
        _model_for_stats = model.module if isinstance(model, nn.DataParallel) else model
        print_model_stats(_model_for_stats, img_size=args.img_size, in_channels=1, device='cuda')
    except Exception as e:
        print(f"⚠️ 模型统计失败: {e}")
    print(f"Train samples: {len(db_train)}, Val volumes: {len(db_val)}")
    print("=" * 60 + "\n")
    # 统计完后让模型回到训练状态
    model.train()

    best_performance = 0.0
    best_epoch = 0
    best_mean_dice = 0.0
    best_mean_hd95 = 0.0
    best_miou = 0.0
    best_acc = 0.0
    best_recall = 0.0

    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        # ==================== 训练阶段 ====================
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            if iter_num % 20 == 0:
                logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

        # ==================== 每个 epoch 结束时验证 ====================
        print(f"\n=== Epoch {epoch_num + 1}/{max_epoch} Validation ===")
        metric_list = 0.0
        seg_metric = SegmentationMetric(num_classes=num_classes)  # 混淆矩阵指标
        model.eval()
        with torch.no_grad():
            for sampled_batch in valloader:
                image, label, case_name = (sampled_batch["image"],
                                           sampled_batch["label"],
                                           sampled_batch['case_name'][0])

                # 原有 Dice / HD95 计算
                metric_i = test_single_volume(image, label, model, classes=args.num_classes,
                                              patch_size=[args.img_size, args.img_size],
                                              test_save_path=None, case=case_name, z_spacing=1)
                metric_list += np.array(metric_i)

                # 新增:计算 mIoU / Accuracy / Recall
                pred_vol = predict_volume_for_metrics(
                    image, model, patch_size=[args.img_size, args.img_size])
                label_np = label.squeeze(0).cpu().numpy().astype(np.int64)
                seg_metric.update(pred_vol, label_np)

        metric_list = metric_list / len(db_val)
        performance = np.mean(metric_list, axis=0)[0]   # Mean Dice
        mean_hd95 = np.mean(metric_list, axis=0)[1]

        summary = seg_metric.summary()
        miou = summary["mIoU"]
        pixel_acc = summary["pixel_acc"]
        mean_recall = summary["mean_recall"]
        mean_precision = summary["mean_precision"]

        # 日志 + 打印
        logging.info(
            'Epoch %d Val | mean_dice: %.4f | mean_hd95: %.4f | mIoU: %.4f | PixelAcc: %.4f | '
            'MeanRecall: %.4f | MeanPrecision: %.4f'
            % (epoch_num + 1, performance, mean_hd95, miou, pixel_acc, mean_recall, mean_precision))
        print(f'Epoch {epoch_num + 1} Validation →')
        print(f'   Mean Dice      : {performance:.4f}')
        print(f'   Mean HD95      : {mean_hd95:.4f}')
        print(f'   mIoU           : {miou:.4f}')
        print(f'   Pixel Accuracy : {pixel_acc:.4f}')
        print(f'   Mean Recall    : {mean_recall:.4f}')
        print(f'   Mean Precision : {mean_precision:.4f}')
        for c in range(num_classes):
            print(f'     - class {c}: IoU={summary["iou_per_class"][c]:.4f} '
                  f'Recall={summary["recall_per_class"][c]:.4f} '
                  f'Precision={summary["precision_per_class"][c]:.4f}')

        # TensorBoard
        writer.add_scalar('val/mean_dice', performance, epoch_num + 1)
        writer.add_scalar('val/mean_hd95', mean_hd95, epoch_num + 1)
        writer.add_scalar('val/mIoU', miou, epoch_num + 1)
        writer.add_scalar('val/pixel_acc', pixel_acc, epoch_num + 1)
        writer.add_scalar('val/mean_recall', mean_recall, epoch_num + 1)
        writer.add_scalar('val/mean_precision', mean_precision, epoch_num + 1)

        # 更新最优模型(以 Dice 作为准则,同时记录其他指标)
        if performance > best_performance:
            best_performance = performance
            best_epoch = epoch_num + 1
            best_mean_dice = performance
            best_mean_hd95 = mean_hd95
            best_miou = miou
            best_acc = pixel_acc
            best_recall = mean_recall
            save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
            torch.save(model.state_dict(), save_mode_path)
            print(f"★★★ Best model updated! Mean Dice = {performance:.4f} (Epoch {best_epoch}) ★★★")

        # 最后一个 epoch 额外保存
        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        model.train()

    # ==================== 训练完全结束时打印最终最优结果 ====================
    print("\n" + "=" * 60)
    print("🎉 TRAINING FINISHED!")
    print(f"Best Epoch      : {best_epoch}")
    print(f"Best Mean Dice  : {best_mean_dice:.4f}")
    print(f"Best Mean HD95  : {best_mean_hd95:.4f}")
    print(f"Best mIoU       : {best_miou:.4f}")
    print(f"Best Pixel Acc  : {best_acc:.4f}")
    print(f"Best Mean Recall: {best_recall:.4f}")
    print(f"Best model saved as: {os.path.join(snapshot_path, 'best_model.pth')}")
    print("=" * 60)

    # ====================== 最优模型可视化 + 每张图带指标 ======================
    print("\n" + "=" * 80)
    print("🚀 开始使用 Best Model 生成【带评价指标】的可视化结果...")
    print("=" * 80)

    best_model_path = os.path.join(snapshot_path, 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"✅ 已加载最佳模型: {best_model_path}")
    else:
        print("⚠️ 未找到 best_model.pth,使用当前模型进行可视化")

    model.eval()
    vis_dir = os.path.join(snapshot_path, 'best_visualization')
    os.makedirs(vis_dir, exist_ok=True)

    metric_list = []
    case_metrics = {}
    final_seg_metric = SegmentationMetric(num_classes=num_classes)

    with torch.no_grad():
        for sampled_batch in tqdm(valloader, desc="Generating Visualization with Metrics"):
            image = sampled_batch["image"].cuda()
            label = sampled_batch["label"]
            case_name = sampled_batch['case_name'][0]

            # 预测(用于可视化的简单推理,可能仅在 2D 情况正确)
            output = model(image)
            pred = torch.argmax(output, dim=1).cpu()

            # Dice / HD95
            metric_i = test_single_volume(
                image.cpu(), label, model, classes=args.num_classes,
                patch_size=[args.img_size, args.img_size],
                test_save_path=None, case=case_name, z_spacing=1
            )
            metric_list.append(metric_i)
            mean_dice = float(np.mean(metric_i, axis=0)[0])
            mean_hd95 = float(np.mean(metric_i, axis=0)[1])

            # mIoU / Acc / Recall(用切片级推理得到完整预测体)
            pred_vol = predict_volume_for_metrics(
                image.cpu(), model, patch_size=[args.img_size, args.img_size])
            label_np = label.squeeze(0).cpu().numpy().astype(np.int64)
            final_seg_metric.update(pred_vol, label_np)

            # 单 case 简易指标(用于命名展示)
            case_metric = SegmentationMetric(num_classes=num_classes)
            case_metric.update(pred_vol, label_np)
            case_sum = case_metric.summary()
            case_metrics[case_name] = {
                "dice": mean_dice,
                "hd95": mean_hd95,
                "miou": case_sum["mIoU"],
                "acc": case_sum["pixel_acc"],
                "recall": case_sum["mean_recall"],
            }

            # -------- 可视化 --------
            if image.shape[1] == 1:
                img_np = image[0, 0].cpu().numpy()
            else:
                img_np = image[0].mean(dim=0).cpu().numpy()
            img_np = img_np.squeeze()

            gt_np = label[0].cpu().numpy().squeeze()
            pred_np = pred[0].numpy().squeeze()

            fig, axs = plt.subplots(1, 4, figsize=(24, 6))

            axs[0].imshow(img_np, cmap='gray')
            axs[0].set_title('Input')
            axs[0].axis('off')

            axs[1].imshow(gt_np, cmap='tab20')
            axs[1].set_title('Ground Truth')
            axs[1].axis('off')

            axs[2].imshow(pred_np, cmap='tab20')
            axs[2].set_title('Prediction')
            axs[2].axis('off')

            overlay = np.stack([img_np, img_np, img_np], axis=-1).astype(np.float32)
            overlay = (overlay / (overlay.max() + 1e-8) * 0.65)
            overlay[gt_np > 0] = overlay[gt_np > 0] + [0.7, 0.0, 0.0]
            overlay[pred_np > 0] = overlay[pred_np > 0] + [0.0, 0.7, 0.0]
            overlay = np.clip(overlay, 0, 1)

            axs[3].imshow(overlay)
            axs[3].set_title(f'Overlay\nDice: {mean_dice:.4f} | HD95: {mean_hd95:.2f}\n'
                             f'mIoU: {case_sum["mIoU"]:.4f} | Acc: {case_sum["pixel_acc"]:.4f} | '
                             f'Recall: {case_sum["mean_recall"]:.4f}')
            axs[3].axis('off')

            plt.suptitle(f'Case: {case_name}   |   Mean Dice: {mean_dice:.4f}', fontsize=16)
            plt.tight_layout()

            save_path = os.path.join(vis_dir, f'{case_name}_Dice{mean_dice:.4f}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

    # ====================== 保存整体指标总结 ======================
    metric_array = np.array(metric_list)
    avg_dice = np.mean(metric_array[:, :, 0])
    avg_hd95 = np.mean(metric_array[:, :, 1])
    final_summary = final_seg_metric.summary()

    print(f"\n🎯 Best Model Validation Result:")
    print(f"   Average Dice    : {avg_dice:.4f}")
    print(f"   Average HD95    : {avg_hd95:.2f}")
    print(f"   mIoU            : {final_summary['mIoU']:.4f}")
    print(f"   Pixel Accuracy  : {final_summary['pixel_acc']:.4f}")
    print(f"   Mean Recall     : {final_summary['mean_recall']:.4f}")
    print(f"   Mean Precision  : {final_summary['mean_precision']:.4f}")
    for c in range(num_classes):
        print(f"     class {c}: IoU={final_summary['iou_per_class'][c]:.4f} "
              f"Recall={final_summary['recall_per_class'][c]:.4f} "
              f"Precision={final_summary['precision_per_class'][c]:.4f}")

    with open(os.path.join(vis_dir, 'metrics_summary.txt'), 'w') as f:
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Average Dice   : {avg_dice:.4f}\n")
        f.write(f"Average HD95   : {avg_hd95:.2f}\n")
        f.write(f"mIoU           : {final_summary['mIoU']:.4f}\n")
        f.write(f"Pixel Accuracy : {final_summary['pixel_acc']:.4f}\n")
        f.write(f"Mean Recall    : {final_summary['mean_recall']:.4f}\n")
        f.write(f"Mean Precision : {final_summary['mean_precision']:.4f}\n")
        for c in range(num_classes):
            f.write(f"  class {c}: IoU={final_summary['iou_per_class'][c]:.4f} "
                    f"Recall={final_summary['recall_per_class'][c]:.4f} "
                    f"Precision={final_summary['precision_per_class'][c]:.4f}\n")
        f.write("\n" + "=" * 60 + "\nPer Case Metrics (sorted by Dice):\n")
        for case, m in sorted(case_metrics.items(), key=lambda x: x[1]["dice"], reverse=True):
            f.write(f"{case:35} Dice: {m['dice']:.4f}  HD95: {m['hd95']:.2f}  "
                    f"mIoU: {m['miou']:.4f}  Acc: {m['acc']:.4f}  Recall: {m['recall']:.4f}\n")

    print(f"✅ 可视化结果已保存到: {vis_dir}")
    print(f"   每张图片文件名已包含 Dice 分数,方便排序。")

    writer.close()
    return "Training Finished!"