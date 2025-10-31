import torch
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from utils import *
import h5py
import torchvision.transforms.functional as TF

def train_model(model, n_epochs, train_loader, valid_loader, optimizer, criterion, device, savepath, writer, print_file):
    train_loss = []
    valid_loss = [] 
    
    train_acc_list = []
    valid_acc_list = []
    
    train_auc_benign = []
    train_auc_indolent = []
    train_auc_aggressive = []
    train_auc_average = []
    valid_auc_benign = []
    valid_auc_indolent = []
    valid_auc_aggressive = []
    valid_auc_average = []

    if n_epochs < 50 :
        print('Since epochs is less than 50, no models will be saved')
        
    for epoch in tqdm(range(1, n_epochs + 1), desc ="Epoch", colour="red"):
        train_pred = []
        train_target = []
        valid_pred = []
        valid_target = []

        model.train(True)
        loss_list = []
        for counter, (data, target) in enumerate(train_loader, 1):
            data = data.to(device).float()
            target = target.squeeze().type(torch.LongTensor).to(device)
            
            output = model(data)
            loss = criterion(output, target)
            loss_list.append(loss.detach().cpu().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss_val = np.average(loss_list)
        writer.add_scalar('Running Loss', train_loss_val, epoch)        

        with torch.no_grad():
            model.eval()
            top1_accuracy = 0
            loss_list = []
            for data, target in valid_loader:
                data = data.to(device).float()
                target = target.squeeze().type(torch.LongTensor).to(device)

                output = model(data)
                top1 = accuracy(output, target, topk=(1,))
                top1_accuracy += top1[0]
                loss = criterion(output, target)
                loss_list.append(loss.detach().cpu().item())
                
                valid_pred.append(output.softmax(dim=1).detach().cpu().numpy())
                valid_target.append(target.detach().cpu().numpy())
            top1_accuracy /= (counter + 1)
            valid_loss.append(np.average(loss_list))
            
            top1_train_accuracy = 0
            loss_list = []
            for counter, (data, target) in enumerate(train_loader):
                data = data.to(device).float()
                target = target.squeeze().type(torch.LongTensor).to(device)

                output = model(data)
                loss = criterion(output, target)
                loss_list.append(loss.detach().cpu().item())
                top1 = accuracy(output, target, topk=(1,))
                top1_train_accuracy += top1[0]
                
                train_pred.append(output.softmax(dim=1).detach().cpu().numpy())
                train_target.append(target.detach().cpu().numpy())
            top1_train_accuracy /= (counter + 1)    
            train_loss.append(np.average(loss_list))
        
            train_pred_ = np.asarray(np.concatenate(train_pred, axis=0), dtype=np.float32)
            train_pred_classes = np.argmax(train_pred_, axis=1)
            train_target_classes = np.asarray(np.concatenate(train_target, axis=0), dtype=np.float32)
            valid_pred_ = np.asarray(np.concatenate(valid_pred, axis=0), dtype=np.float32)
            valid_pred_classes = np.argmax(valid_pred_, axis=1)
            valid_target_classes = np.asarray(np.concatenate(valid_target, axis=0), dtype=np.float32)
            
            train_acc = accuracy_score(train_target_classes, train_pred_classes)
            valid_acc = accuracy_score(valid_target_classes, valid_pred_classes)
            
            train_acc_list.append(train_acc)
            valid_acc_list.append(valid_acc)
            
            train_auc_classes = compute_OvR_AUC(true_labels=train_target_classes.reshape(-1, 1), logits=train_pred_.reshape(-1, 3), num_classes=3)
            valid_auc_classes = compute_OvR_AUC(true_labels=valid_target_classes.reshape(-1, 1), logits=valid_pred_.reshape(-1, 3), num_classes=3)
            
            train_auc_benign.append(train_auc_classes[0])
            train_auc_indolent.append(train_auc_classes[1])
            train_auc_aggressive.append(train_auc_classes[2])
            train_auc_average.append(np.mean(train_auc_classes))
            valid_auc_benign.append(valid_auc_classes[0])
            valid_auc_indolent.append(valid_auc_classes[1])
            valid_auc_aggressive.append(valid_auc_classes[2])
            valid_auc_average.append(np.mean(valid_auc_classes))

            print_msg = (f'Epoch:{epoch} \t'
                         f'Train OvR AUC: {train_auc_classes}\t'
                         f'Test OvR AUC: {valid_auc_classes}\t'
                         f'T_Loss: {train_loss[-1]:.5f} \t'
                         f'T_ACC: {train_acc:.5f} \t'
                         f'train top1 accuracy:{top1_train_accuracy}\t'
                         f'V_Loss: {valid_loss[-1]:.5f} \t'
                         f'V_ACC: {valid_acc:.5f} \t'
                         f'test top1 accuracy:{top1_accuracy}\t')
            print(print_msg, file=print_file)

            writer.add_scalar('Loss/Training Loss', train_loss[-1], epoch)
            writer.add_scalar('Loss/Validation Loss', valid_loss[-1], epoch)
            writer.add_scalar('Loss/Training Accuracy', train_acc*100, epoch)
            writer.add_scalar('Loss/Validation Accuracy', valid_acc*100, epoch)
            writer.add_scalar('Loss/Training AUC - Benign v Rest', train_auc_classes[0], epoch)
            writer.add_scalar('Loss/Validation AUC - Benign v Rest', valid_auc_classes[0], epoch)
            writer.add_scalar('Loss/Training AUC - Indolent v Rest', train_auc_classes[1], epoch)
            writer.add_scalar('Loss/Validation AUC - Indolent v Rest', valid_auc_classes[1], epoch)
            writer.add_scalar('Loss/Training AUC - Aggressive v Rest', train_auc_classes[2], epoch)
            writer.add_scalar('Loss/Validation AUC - Aggressive v Rest', valid_auc_classes[2], epoch)
            writer.add_scalar('Loss/Training AUC - Mean', np.mean(train_auc_classes), epoch)
            writer.add_scalar('Loss/Validation AUC - Mean', np.mean(valid_auc_classes), epoch)
            
            if epoch == 50:
                best_acc_valid = valid_acc
                best_acc_train = train_acc
                state = {'epoch': epoch, 'state_dict': model.state_dict(), 'train_acc': train_acc,
                    'valid_acc' : valid_acc, 'optimizer' : optimizer.state_dict()}
                torch.save(state, savepath+'/models/best_model_acc_valid.pt')
                torch.save(state, savepath+'/models/best_model_acc_train.pt')
            
            if epoch > 50:
                is_best_valid_acc =  valid_acc >= best_acc_valid
                is_best_train_acc = train_acc >= best_acc_train
                if is_best_valid_acc:
                    best_acc_valid = valid_acc
                    state = {'epoch': epoch, 'state_dict': model.state_dict(), 'train_acc': train_acc,
                        'valid_acc' : valid_acc, 'optimizer' : optimizer.state_dict()}
                    torch.save(state, savepath+'/models/best_model_acc_valid.pt')
                    
                if is_best_train_acc:
                    best_acc_train = train_acc
                    state = {'epoch': epoch, 'state_dict': model.state_dict(), 'train_acc': train_acc,
                        'valid_acc' : valid_acc, 'optimizer' : optimizer.state_dict()}
                    torch.save(state, savepath+'/models/best_model_acc_train.pt')
            
    torch.save(model.state_dict(), savepath+'/models/model.pt')
    torch.save(optimizer.state_dict(), savepath+'/models/optimizer.pt')

    return  train_acc, valid_acc, best_acc_train, best_acc_valid, train_auc_classes, valid_auc_classes
