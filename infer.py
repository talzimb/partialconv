import os.path as osp
import torch
import time
from grad_cam.grad_cam import GradCAM
from grad_cam.utils import get_prediction_strings, save_images
from utils.tsne_analysis import tsne


def inference(val_loader, model, criterion, args, root):
    # switch to evaluate mode
    model.eval()

    # create gradcam object
    visualizer = GradCAM(model=model)
    all_confidence_scores, all_prediction_strings = [], []

    last_layers = []
    targets = []
    predictions = []

    with torch.no_grad():
        end = time.time()
        for i, (input, target, mask, file_names) in enumerate(val_loader):
            torch.cuda.empty_cache()
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            with torch.enable_grad():
                input = input.permute(0, 3, 1, 2)
                output, last_fc = model(input, mask)
                # loss = criterion(output, target)
                loss = torch.gather(output, dim=1, index=target.unsqueeze(-1)).sum()
                loss.backward(retain_graph=True)

            # collect last layesr for tsne
            last_layers.append(last_fc)
            targets.append(target)
            predictions.append(torch.argmax(output, dim=1))

            # Generate and save the GracCAM visualization maps
            prediction_strings = get_prediction_strings(output, target)
            heatmap_images, overlap_scores = visualizer.get_heatmap_projection(file_names)

            # Generate confidence scores
            confidence_scores = (output.max(axis=1).values.detach().cpu().numpy() + overlap_scores) / 2

            save_images(heatmap_images,
                        [f"{osp.basename(file_names[j])}_{prediction_strings[j]}_{confidence_scores[j]:.4g}.jpg" \
                         for j in range(len(file_names))],
                        osp.join(root, args.ckptdirprefix, "inference_results", "gc_out"))
            all_confidence_scores.extend(confidence_scores)
            all_prediction_strings.extend(prediction_strings)

        # convert outputs to tensors
        results = torch.cat(predictions).cpu(), torch.cat(targets).cpu()
        last_layers = torch.cat(last_layers).cpu()
        # create tsne analysis
        tsne(feature_map=last_layers, results=results, component_num=2,
             dir_path=osp.join(args.ckptdirprefix, "inference_results"))

        # generate_confidence_histogram(all_confidence_scores, all_prediction_strings,
        #                               osp.join(args.ckptdirprefix, "inference_results", "confidence_histogram.png"))

    #         # measure accuracy and record loss
    #         # prec1, prec5 = accuracy(output, target, topk=(1, 5))
    #         losses.update(loss.item(), input.size(0))
    #         # top1.update(prec1[0], input.size(0))
    #         # top5.update(prec5[0], input.size(0))
    #
    #         # measure elapsed time
    #         batch_time.update(time.time() - end)
    #         end = time.time()
    #
    #         if i % args.print_freq == 0:
    #             print('Test: [{0}/{1}]\t'
    #                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
    #                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
    #                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
    #                    i, len(val_loader), batch_time=batch_time, loss=losses,
    #                    top1=top1, top5=top5))
    #
    #             with open(args.logger_fname, "a") as log_file:
    #                 log_file.write('Test: [{0}/{1}]\t'
    #                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
    #                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
    #                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
    #                    i, len(val_loader), batch_time=batch_time, loss=losses,
    #                    top1=top1, top5=top5))
    #
    #     print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
    #           .format(top1=top1, top5=top5))
    #
    #     with open(args.logger_fname, "a") as final_log_file:
    #         final_log_file.write(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
    #           .format(top1=top1, top5=top5))
    #
    # return top1.avg
