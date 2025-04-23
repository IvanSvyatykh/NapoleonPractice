import sys
import time
import random
from clearml import Task
import torch
import os
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import yaml
import pandas as pd
from prepare_dataset import add_csv_to_img_dir
from torch.amp import autocast, GradScaler
from argparse import ArgumentParser
from pathlib import Path
from utils import CTCLabelConverter, AttnLabelConverter, Averager, AttrDict
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from validate import validation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_config(file_path: Path) -> AttrDict:
    with open(file_path, "r", encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    if opt.lang_char == "None":
        characters = ""
        for data in opt["select_data"].split("-"):
            csv_path = os.path.join(opt["train_data"], data, "labels.csv")
            df = pd.read_csv(
                csv_path,
                sep="^([^,]+),",
                engine="python",
                usecols=["filename", "words"],
                keep_default_na=False,
            )
            all_char = "".join(df["words"])
            characters += "".join(set(all_char))
        characters = sorted(set(characters))
        opt.character = "".join(characters)
    else:
        opt.character = opt.number + opt.symbol + opt.lang_char
    os.makedirs(f"./saved_models/{opt.experiment_name}", exist_ok=True)
    return opt


def count_parameters(model: Model) -> int:
    print("Modules, Parameters")
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        # table.add_row([name, param])
        total_params += param
        print(name, param)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def load_model(config: AttrDict) -> Model:
    model = Model(config)
    if config.saved_model != "":
        pretrained_dict = torch.load(config.saved_model)
        if config.new_prediction:
            model.Prediction = nn.Linear(
                model.SequenceModeling_output,
                len(pretrained_dict["module.Prediction.weight"]),
            )

        model = torch.nn.DataParallel(model).to(device)
        print(f"loading pretrained model from {config.saved_model}")
        if config.FT:
            model.load_state_dict(pretrained_dict, strict=False)
        else:
            model.load_state_dict(pretrained_dict)
        if config.new_prediction:
            model.module.Prediction = nn.Linear(
                model.module.SequenceModeling_output, config.num_class
            )
            for name, param in model.module.Prediction.named_parameters():
                if "bias" in name:
                    init.constant_(param, 0.0)
                elif "weight" in name:
                    init.kaiming_normal_(param)
            model = model.to(device)
    else:
        # weight initialization
        for name, param in model.named_parameters():
            if "localization_fc2" in name:
                print(f"Skip {name} as it is already initialized")
                continue
            try:
                if "bias" in name:
                    init.constant_(param, 0.0)
                elif "weight" in name:
                    init.kaiming_normal_(param)
            except Exception as e:  # for batchnorm.
                if "weight" in name:
                    param.data.fill_(1)
                continue
        model = torch.nn.DataParallel(model).to(device)
    return model


def create_converter(config: AttrDict) -> CTCLabelConverter | AttnLabelConverter:
    if "CTC" in config.Prediction:
        converter = CTCLabelConverter(config.character)
    else:
        converter = AttnLabelConverter(config.character)
    return converter


def create_criterion(config: AttrDict) -> torch.nn.CTCLoss | torch.nn.CrossEntropyLoss:
    if "CTC" in config.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    return criterion


def create_optimizer(config: AttrDict, params: list) -> torch.optim.optimizer.Optimizer:
    if config.optim == "adam":
        optimizer = optim.Adam(params, lr=config.lr, eps=config.eps)
    else:
        optimizer = optim.Adadelta(params, lr=config.lr, rho=config.rho, eps=config.eps)
    return optimizer


def freeze_params(config: AttrDict, model: Model) -> Model:
    try:
        if config.freeze_FeatureFxtraction:
            for param in model.module.FeatureExtraction.parameters():
                param.requires_grad = False
        if config.freeze_SequenceModeling:
            for param in model.module.SequenceModeling.parameters():
                param.requires_grad = False
    except:
        raise Exception("Can not freeze model params")
    return model


def write_config_to_file(config: AttrDict) -> None:
    with open(
        f"./saved_models/{config.experiment_name}/opt.txt", "a", encoding="utf8"
    ) as config_file:
        opt_log = "------------ Options -------------\n"
        args = vars(config)
        for k, v in args.items():
            opt_log += f"{str(k)}: {str(v)}\n"
        opt_log += "---------------------------------------\n"
        print(opt_log)
        config_file.write(opt_log)



def train_epoch(model:Model,)

def train(
    config: AttrDict, task: Task, show_number: int = 2, amp: bool = False
) -> None:
    """dataset preparation"""
    if not config.data_filtering_off:
        print(
            "Filtering the images containing characters which are not in opt.character"
        )
        print("Filtering the images whose label is longer than opt.batch_max_length")

    config.select_data = config.select_data.split("-")
    config.batch_ratio = config.batch_ratio.split("-")
    train_dataset = Batch_Balanced_Dataset(config)

    log = open(
        f"./saved_models/{config.experiment_name}/log_dataset.txt", "a", encoding="utf8"
    )
    AlignCollate_valid = AlignCollate(
        imgH=config.imgH,
        imgW=config.imgW,
        keep_ratio_with_pad=config.PAD,
        contrast_adjust=config.contrast_adjust,
    )
    valid_dataset, valid_dataset_log = hierarchical_dataset(
        root=config.valid_data, opt=config
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=min(32, config.batch_size),
        shuffle=True,
        num_workers=int(config.workers),
        prefetch_factor=512,
        collate_fn=AlignCollate_valid,
        pin_memory=True,
    )
    log.write(valid_dataset_log)
    print("-" * 80)
    log.write("-" * 80 + "\n")
    log.close()

    """ model configuration """
    converter = create_converter(config)
    config.num_class = len(converter.character)

    if config.rgb:
        config.input_channel = 3
    model = load_model(config)
    model.train()

    """ setup loss """
    criterion = create_criterion(config)
    loss_avg = Averager()

    freeze_params(config, model)
    filtered_parameters = [
        p for p in filter(lambda p: p.requires_grad, model.parameters())
    ]
    optimizer = create_optimizer(config, filtered_parameters)

    write_config_to_file(config)

    """ start training """
    start_iter = 0
    if config.saved_model != "":
        try:
            start_iter = int(config.saved_model.split("_")[-1].split(".")[0])
            print(f"continue to train, start_iter: {start_iter}")
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    i = start_iter

    scaler = GradScaler()
    t1 = time.time()

    while True:
        optimizer.zero_grad(set_to_none=True)

        if amp:
            with autocast():
                image_tensors, labels = train_dataset.get_batch()
                image = image_tensors.to(device)
                text, length = converter.encode(
                    labels, batch_max_length=config.batch_max_length
                )
                batch_size = image.size(0)

                if "CTC" in config.Prediction:
                    preds = model(image, text).log_softmax(2)
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    preds = preds.permute(1, 0, 2)
                    torch.backends.cudnn.enabled = False
                    cost = criterion(
                        preds, text.to(device), preds_size.to(device), length.to(device)
                    )
                    torch.backends.cudnn.enabled = True
                else:
                    preds = model(image, text[:, :-1])  # align with Attention.forward
                    target = text[:, 1:]  # without [GO] Symbol
                    cost = criterion(
                        preds.view(-1, preds.shape[-1]), target.contiguous().view(-1)
                    )
            scaler.scale(cost).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            image_tensors, labels = train_dataset.get_batch()
            image = image_tensors.to(device)
            text, length = converter.encode(
                labels, batch_max_length=config.batch_max_length
            )
            batch_size = image.size(0)
            if "CTC" in config.Prediction:
                preds = model(image, text).log_softmax(2)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                preds = preds.permute(1, 0, 2)
                torch.backends.cudnn.enabled = False
                cost = criterion(
                    preds, text.to(device), preds_size.to(device), length.to(device)
                )
                torch.backends.cudnn.enabled = True
            else:
                preds = model(image, text[:, :-1])  # align with Attention.forward
                target = text[:, 1:]  # without [GO] Symbol
                cost = criterion(
                    preds.view(-1, preds.shape[-1]), target.contiguous().view(-1)
                )
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        loss_avg.add(cost)

        # validation part
        if (i % config.valInterval == 0) and (i != 0):
            print("training time: ", time.time() - t1)
            t1 = time.time()
            elapsed_time = time.time() - start_time
            # for log
            with open(
                f"./saved_models/{config.experiment_name}/log_train.txt",
                "a",
                encoding="utf8",
            ) as log:
                model.eval()
                with torch.no_grad():
                    (
                        valid_loss,
                        current_accuracy,
                        current_norm_ED,
                        preds,
                        confidence_score,
                        labels,
                        infer_time,
                        length_of_data,
                    ) = validation(
                        model, criterion, valid_loader, converter, config, device
                    )
                model.train()

                # training loss and validation loss
                train_loss = loss_avg.val()
                loss_log = f"[{i}/{config.num_iter}] Train loss: {train_loss:0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}"
                loss_avg.reset()

                task.get_logger().report_scalar(
                    "Train loss", "iteration", train_loss, iteration=i
                )
                task.get_logger().report_scalar(
                    "Val loss", "iteration", valid_loss, iteration=i
                )
                current_model_log = f"{'Current_accuracy':17s}: {current_accuracy:0.3f}, {'Current_norm_ED':17s}: {current_norm_ED:0.4f}"

                # keep best accuracy model (on valid dataset)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(
                        model.state_dict(),
                        f"./saved_models/{config.experiment_name}/best_accuracy.pth",
                    )
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(
                        model.state_dict(),
                        f"./saved_models/{config.experiment_name}/best_norm_ED.pth",
                    )
                best_model_log = f"{'Best_accuracy':17s}: {best_accuracy:0.3f}, {'Best_norm_ED':17s}: {best_norm_ED:0.4f}"
                task.get_logger().report_scalar(
                    "Val accuracy", "iterations", best_accuracy, iteration=i
                )
                loss_model_log = f"{loss_log}\n{current_model_log}\n{best_model_log}"
                print(loss_model_log)
                log.write(loss_model_log + "\n")

                # show some predicted results
                dashed_line = "-" * 80
                head = f"{'Ground Truth':25s} | {'Prediction':25s} | Confidence Score & T/F"
                predicted_result_log = f"{dashed_line}\n{head}\n{dashed_line}\n"

                # show_number = min(show_number, len(labels))

                start = random.randint(0, len(labels) - show_number)
                for gt, pred, confidence in zip(
                    labels[start : start + show_number],
                    preds[start : start + show_number],
                    confidence_score[start : start + show_number],
                ):
                    if "Attn" in config.Prediction:
                        gt = gt[: gt.find("[s]")]
                        pred = pred[: pred.find("[s]")]

                    predicted_result_log += f"{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n"
                predicted_result_log += f"{dashed_line}"
                print(predicted_result_log)
                log.write(predicted_result_log + "\n")
                print("validation time: ", time.time() - t1)
                t1 = time.time()

        if i == config.num_iter:
            print("end the training")
            sys.exit()
        i += 1


def main(path_to_conf: Path) -> None:
    assert path_to_conf.exists()
    config: AttrDict = get_config(path_to_conf)
    task = Task.init(project_name="retail/ocr/easyocr", task_name=config.task_name)
    add_csv_to_img_dir(Path(config.train_data), config.train_metadata_file_name)
    add_csv_to_img_dir(Path(config.valid_data), config.val_metadata_file_name)
    print(f"Start train on {device}")
    train(config=config, task=task)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path_to_conf", "-conf", required=True, type=Path)
    args = parser.parse_args()
    main(args.path_to_conf)
