import os
import shutil
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import DataLoader

from .cgcnn.data import CIFData
from .cgcnn.data import collate_pool
from .cgcnn.model import CrystalGraphConvNet


def predict_cgcnn(cifpath, modelpath=None, task='regression', batch_size=256,
                  workers=0, disable_cuda=False, print_freq=10, depth=-1,
                  save_results=False, output_file='results.csv'):
    """
    API function to make predictions using CGCNN model

    Args:
        cifpath: Path to the directory of CIF files
        modelpath: Path to the trained model
        task: 'regression' or 'classification' (default: regression)
        batch_size: Mini-batch size (default: 256)
        workers: Number of data loading workers (default: 0)
        disable_cuda: Whether to disable CUDA (default: False)
        print_freq: Print frequency (default: 10)
        depth: Threshold depth for surface atoms, -1 to ignore (default: -1)
        save_results: Whether to save results to CSV (default: False)
        output_file: Output CSV file name (default: 'test_results.csv')

    Returns:
        Dictionary containing predictions, targets, and metrics
    """
    # Initialize args like in original code
    if not modelpath:
        script_dir = Path(__file__).parent
        modelpath = script_dir / "./models/model_best.pth.tar"

    args = type('Args', (), {})()
    args.cifpath = cifpath
    args.modelpath = modelpath
    args.task = task
    args.batch_size = batch_size
    args.workers = workers
    args.disable_cuda = disable_cuda
    args.print_freq = print_freq
    args.depth = depth
    args.cuda = not args.disable_cuda and torch.cuda.is_available()

    # Load model checkpoint
    if os.path.isfile(args.modelpath):
        print("=> loading model params '{}'".format(args.modelpath))
        model_checkpoint = torch.load(args.modelpath,
                                      map_location=lambda storage, loc: storage,
                                      weights_only=False)  # Add weights_only parameter
        model_args_dict = model_checkpoint['args']
        print("=> loaded model params '{}'".format(args.modelpath))
    else:
        raise FileNotFoundError("=> no model params found at '{}'".format(args.modelpath))

    # Create a simple object from the args dictionary
    class ModelArgs:
        def __init__(self, args_dict):
            for key, value in args_dict.items():
                setattr(self, key, value)

    model_args = ModelArgs(model_args_dict)

    if model_args.task == 'regression':
        best_mae_error = 1e10
    else:
        best_mae_error = 0.

    # Load data
    dataset = CIFData(args.cifpath, depth=args.depth)
    collate_fn = collate_pool
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.workers, collate_fn=collate_fn,
                             pin_memory=args.cuda)

    sample_data_list = [dataset[i] for i in range(len(dataset))]
    _, targets_dict, _ = collate_pool(sample_data_list)

    normalizers = {}
    for i in range(len(targets_dict)):
        normalizers[f"normalizer{i + 1}"] = Normalizer(targets_dict[i])

    # Build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=model_args.atom_fea_len,
                                n_conv=model_args.n_conv,
                                h_fea_len=model_args.h_fea_len,
                                n_h=1,
                                classification=True if model_args.task == 'classification' else False,
                                i_tasks=targets_dict.keys())
    if args.cuda:
        model.cuda()

    # Define loss function
    if model_args.task == 'classification':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()

    # Load model weights
    if os.path.isfile(args.modelpath):
        checkpoint = torch.load(args.modelpath,
                                map_location=lambda storage, loc: storage,
                                weights_only=False)  # Add weights_only parameter
        model.load_state_dict(checkpoint['state_dict'])

        for i in range(len(targets_dict)):
            normalizer_key = f"normalizer{i + 1}"
            if normalizer_key in checkpoint:
                normalizers[normalizer_key].load_state_dict(checkpoint[normalizer_key])

    else:
        raise FileNotFoundError("=> no model found at '{}'".format(args.modelpath))

    # Validate and get results
    results = validate(test_loader, model, criterion, args, model_args,
                       test=True, save_results=save_results,
                       output_file=output_file, **normalizers)

    return results


def validate(val_loader, model, criterion, args, model_args, test=False,
             save_results=False, output_file='test_results.csv', **normalizers):
    """
    Validate or test the model

    Modified version of original validate function that returns results
    """
    n_tasks = len(model.i_tasks)
    batch_time = AverageMeter()
    losses = [AverageMeter() for _ in range(n_tasks)]
    losses_total = AverageMeter()

    if model_args.task == 'regression':
        mae_errors = [AverageMeter() for _ in range(n_tasks)]
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    if test:
        test_targets = [[] for _ in range(n_tasks)]
        test_preds = [[] for _ in range(n_tasks)]
        test_cif_ids = []

    # Switch to evaluate mode
    model.eval()

    end = time.time()
    normalizer_list = list(normalizers.values())

    for batch_idx, (input_data, targets, batch_cif_ids) in enumerate(val_loader):
        with torch.no_grad():
            if args.cuda:
                input_var = (
                    Variable(input_data[0].cuda(non_blocking=True)),
                    Variable(input_data[1].cuda(non_blocking=True)),
                    input_data[2].cuda(non_blocking=True),
                    [crys_idx.cuda(non_blocking=True) for crys_idx in input_data[3]]
                )
            else:
                input_var = (Variable(input_data[0]),
                             Variable(input_data[1]),
                             input_data[2],
                             input_data[3])

        if model_args.task == 'regression':
            target_vars = []
            for i, count in enumerate(model.i_tasks):
                target_normed = normalizer_list[i].norm(targets[count])
                if args.cuda:
                    target_vars.append(Variable(target_normed.cuda(non_blocking=True)))
                else:
                    target_vars.append(Variable(target_normed))
        else:
            target_normed = targets.view(-1).long()
            if args.cuda:
                target_var = Variable(target_normed.cuda(non_blocking=True))
            else:
                target_var = Variable(target_normed)

        # Compute output
        outputs = model(*input_var)

        if model_args.task == 'regression':
            loss = []
            for i in range(n_tasks):
                loss.append(criterion(outputs[i], target_vars[i]))
            total_loss = sum(loss)
        else:
            loss = criterion(outputs, target_var)
            total_loss = loss

        # Measure accuracy and record loss
        if model_args.task == 'regression':
            for i, count in enumerate(model.i_tasks):
                mae_error_val = mae(
                    normalizer_list[i].denorm(outputs[i].data.cpu()),
                    targets[count]
                )
                losses[i].update(loss[i].data.cpu(), targets[count].size(0))
                mae_errors[i].update(mae_error_val, targets[count].size(0))

            losses_total.update(total_loss.data.cpu(), targets[0].size(0))

            if test:
                for i, count in enumerate(model.i_tasks):
                    test_pred = normalizer_list[i].denorm(outputs[i].data.cpu())
                    test_preds[i] += test_pred.view(-1).tolist()
                    test_targets[i] += targets[count].view(-1).tolist()
                test_cif_ids += batch_cif_ids

        else:
            accuracy, precision, recall, fscore, auc_score = class_eval(
                outputs.data.cpu(), target_normed
            )
            losses_total.update(loss.data.cpu().item(), target_normed.size(0))
            accuracies.update(accuracy, target_normed.size(0))
            precisions.update(precision, target_normed.size(0))
            recalls.update(recall, target_normed.size(0))
            fscores.update(fscore, target_normed.size(0))
            auc_scores.update(auc_score, target_normed.size(0))

            if test:
                test_pred = torch.exp(outputs.data.cpu())
                test_target = target_normed
                test_preds[0] += test_pred[:, 1].tolist() if test_pred.shape[1] == 2 else test_pred.tolist()
                test_targets[0] += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # Save results if requested
    results_dict = {}
    if test:
        if save_results:
            import csv

            test_targets_preds = []
            for i in range(n_tasks):
                test_targets_preds.append(test_targets[i])
                test_targets_preds.append(test_preds[i])

            with open(output_file, 'w') as f:
                writer = csv.writer(f)
                for values in zip(test_cif_ids, *test_targets_preds):
                    writer.writerow(values)
            print(f"=> results saved to '{output_file}'")

        # Prepare results dictionary
        results_dict = {
            'cif_ids': test_cif_ids,
            'targets': test_targets,
            'predictions': test_preds
        }

        if model_args.task == 'regression':
            metrics_dict = {}
            for i in range(n_tasks):
                metrics_dict[f'MAE{i + 1}'] = mae_errors[i].avg
                metrics_dict[f'Loss{i + 1}'] = losses[i].avg
            metrics_dict['total_loss'] = losses_total.avg
            results_dict['metrics'] = metrics_dict

    return results_dict


class Normalizer(object):
    """Normalize a Tensor and restore it later."""

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
