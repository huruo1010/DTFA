import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_lr
import numpy as np
from PIL import Image


def fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Epoch, cuda, save_period, res_model, det_model, R_criterion, D_criterion,
                 ):
    loss = 0
    val_loss = 0
    loss_s = 0
    x = 0
    R_Consistancy_loss = 0
    D_Consistancy_loss = 0

    model_train.train()
    R_criterion.train()
    D_criterion.train()
    print('Start Train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            images, targets, clearimgs = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                    clearimgs = torch.from_numpy(clearimgs).type(torch.FloatTensor).cuda()
                else:
                    images = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                    clearimgs = torch.from_numpy(clearimgs).type(torch.FloatTensor)

            optimizer.zero_grad()

            # outputs         = model_train(images)
            outputs = model_train(images)

            with torch.no_grad():

                res_model.eval()
                res_out = res_model(clearimgs)

                det_model.eval()
                det_out = det_model(clearimgs)


            loss_value_all = 0

            loss_value = yolo_loss(outputs[0], targets)

            loss_save = loss_value

            R_loss_consistancy = R_criterion((outputs[1][1],outputs[1][0]), (res_out[1],res_out[0]))
            D_loss_consistancy = D_criterion((outputs[1][1],outputs[1][0]), (det_out[1],det_out[0]))
            loss_value = 0.2 * loss_value + R_loss_consistancy + D_loss_consistancy

            loss_value.backward()
            optimizer.step()

            loss += loss_value.item()
            R_Consistancy_loss += R_loss_consistancy.item()
            D_Consistancy_loss += D_loss_consistancy.item()
            loss_s += loss_save.item()

            pbar.set_postfix(**{'loss': loss / (iteration + 1),
                                'R-Consistancy_loss': R_Consistancy_loss / (iteration + 1),
                                'D-Consistancy_loss': D_Consistancy_loss / (iteration + 1),
                                'Detection loss': loss_s / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    images = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]

                optimizer.zero_grad()

                outputs = model_train(images)

                loss_value = yolo_loss(outputs[0], targets)

            val_loss += loss_value.item()
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')

    loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val, R_Consistancy_loss / epoch_step,
                             D_Consistancy_loss / epoch_step, loss_s / epoch_step)
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(model.state_dict(),
                   'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val))

