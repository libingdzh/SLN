import os
from dataset import (
    Dataset_img_clinical, Dataset_img_onlyclinical,
    Dataset_imgonly, Dataset_imgonlyMR, Dataset_imgonlyMG,
    Dataset_imgonlyUS, Dataset_imgonlyMRMG, Dataset_imgonlyMRUS,
    Dataset_imgonlyMGUS, Dataset_img_clinical_notdistance,
    Dataset_img_onlyclinicalnotdistance
)
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import roc_auc_score, accuracy_score
import argparse
import random

from Multimodal_Fusion_Network import (
    MFNet_img_pathology_transcat_l3m_orgTransformer_1,
    MFNet_img_pathology_transcat_l3m_orgTransformer,
    MFNet_img_pathology_transcat_l3m_convnext,
    MFNet_img_pathology_transcat_l3m_densenet121,
    MFNet_img_pathology_transcat_l3m_resnet50,
    MFNet_img_pathology_transcat_l3m_vgg19,
    MFNet_img_pathology_transcat_l3m_onlyGCViT,
    MFNet_img_pathology_transcat_l3m_orgTransformer_1_notdistance,
    MFNet_img_onlypathology_transcat_l3m,
    MFNet_img_onlypathologynotdistance_transcat_l3m,
    MFNet_imgonly_new,
    MFNet_imgonlyMR_new,
    MFNet_imgonlyMG_new,
    MFNet_imgonlyUS_new,
    MFNet_imgonlyMRMG_new,
    MFNet_imgonlyMRUS_new,
    MFNet_imgonlyMGUS_new
)


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Unified Training and Testing Script')

    # Basic parameters
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'test_external'],
                        help='mode: train, test, test_external')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device to use')
    parser.add_argument('--bs', type=int, default=4, help='batch size')
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--dropout', type=float, default=0, help='dropout rate')

    # Model configuration
    parser.add_argument('--model', type=str, default='MAT',
                        choices=['MAT', 'orgTrans', 'convnext', 'densenet121', 'resnet50',
                                 'vgg19', 'onlyGCViT', 'MAT_notdistance', 'onlyCP',
                                 'onlyCP_notdistance', 'imgonly', 'imgonlyMR', 'imgonlyMG',
                                 'imgonlyUS', 'imgonlyMRMG', 'imgonlyMRUS', 'imgonlyMGUS'],
                        help='model architecture')
    parser.add_argument('--name', type=str, default='pathology_cat_l3m', help='network name')
    parser.add_argument('--date', type=str, default='0823', help='experiment date identifier')

    # Data configuration
    parser.add_argument('--data_dir', type=str, default='../data_incohort', help='data directory')
    parser.add_argument('--split_type', type=str, default='ACorADC_1',
                        choices=['ACorADC_1', 'DPA', 'DPA_LIQ', 'DPA_LOQ', 'DPA_UIQ', 'DPA_UOQ',
                                 'DPNA', 'DPNA_LIQ', 'DPNA_LOQ', 'DPNA_UIQ', 'DPNA_UOQ'],
                        help='data split file type')
    parser.add_argument('--dataset_type', type=str, default='allimgclinical',
                        choices=['allimgclinical', 'allimgclinical_notdistance', 'onlyclinical',
                                 'onlyclinical_notdistance', 'imgonly', 'imgonlyMR',
                                 'imgonlyMG', 'imgonlyUS', 'imgonlyMRMG', 'imgonlyMRUS', 'imgonlyMGUS'],
                        help='dataset type')

    # External validation
    parser.add_argument('--excohort', type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7],
                        help='external cohort number (1-7)')

    # Training parameters
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00005, help='weight decay')
    parser.add_argument('--loss_threshold', type=float, default=0.0, help='loss threshold')
    parser.add_argument('--pretrained_path', type=str, default=None, help='pretrained model path')

    args = parser.parse_args()

    # Generate split file name from split_type
    args.split_file = f'{args.data_dir}/output_data_{args.split_type}.pkl'

    return args


def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset(data_dir, split_path, dataset_type, data_set, augment=False):
    """Get the appropriate dataset class based on dataset_type"""
    dataset_map = {
        'allimgclinical': Dataset_img_clinical,
        'allimgclinical_notdistance': Dataset_img_clinical_notdistance,
        'onlyclinical': Dataset_img_onlyclinical,
        'onlyclinical_notdistance': Dataset_img_onlyclinicalnotdistance,
        'imgonly': Dataset_imgonly,
        'imgonlyMR': Dataset_imgonlyMR,
        'imgonlyMG': Dataset_imgonlyMG,
        'imgonlyUS': Dataset_imgonlyUS,
        'imgonlyMRMG': Dataset_imgonlyMRMG,
        'imgonlyMRUS': Dataset_imgonlyMRUS,
        'imgonlyMGUS': Dataset_imgonlyMGUS,
    }
    DatasetClass = dataset_map.get(dataset_type, Dataset_img_clinical)
    return DatasetClass(data_dir, split_path, data_set=data_set, augment=augment)


def get_model(args, num_class=2):
    """Get the appropriate model architecture based on model type"""
    model_map = {
        'MAT': MFNet_img_pathology_transcat_l3m_orgTransformer_1,
        'orgTrans': MFNet_img_pathology_transcat_l3m_orgTransformer,
        'convnext': MFNet_img_pathology_transcat_l3m_convnext,
        'densenet121': MFNet_img_pathology_transcat_l3m_densenet121,
        'resnet50': MFNet_img_pathology_transcat_l3m_resnet50,
        'vgg19': MFNet_img_pathology_transcat_l3m_vgg19,
        'onlyGCViT': MFNet_img_pathology_transcat_l3m_onlyGCViT,
        'MAT_notdistance': MFNet_img_pathology_transcat_l3m_orgTransformer_1_notdistance,
        'onlyCP': MFNet_img_onlypathology_transcat_l3m,
        'onlyCP_notdistance': MFNet_img_onlypathologynotdistance_transcat_l3m,
        'imgonly': MFNet_imgonly_new,
        'imgonlyMR': MFNet_imgonlyMR_new,
        'imgonlyMG': MFNet_imgonlyMG_new,
        'imgonlyUS': MFNet_imgonlyUS_new,
        'imgonlyMRMG': MFNet_imgonlyMRMG_new,
        'imgonlyMRUS': MFNet_imgonlyMRUS_new,
        'imgonlyMGUS': MFNet_imgonlyMGUS_new,
    }
    ModelClass = model_map.get(args.model, MFNet_img_pathology_transcat_l3m_orgTransformer_1)
    return ModelClass(n_classes=num_class, no_cuda=False, drop=args.dropout).cuda()


def get_data_loaders(args, is_external=False):
    """Create data loaders for training, validation, and testing"""
    if is_external:
        # External validation dataset
        data_dir = f'../data_excohort{args.excohort}'
        split_path = f'../data_excohort{args.excohort}/output_excohort{args.excohort}_data_ACorADC_1.pkl'
        save_dir = f'trained_models_excohort{args.excohort}/{args.name}_{args.split_type}/bs{args.bs}_epoch{args.epoch}_seed{args.seed}_{args.date}_drop{args.dropout}'

        test_data = get_dataset(data_dir, split_path, args.dataset_type, 'test', augment=False)
        test_loader = DataLoader(dataset=test_data, batch_size=args.bs, shuffle=False,
                                 num_workers=4, pin_memory=True, drop_last=False)
        return None, None, test_loader, save_dir
    else:
        # Internal training/validation dataset
        save_dir = f'trained_models/{args.name}_{args.split_type}/bs{args.bs}_epoch{args.epoch}_seed{args.seed}_{args.date}_drop{args.dropout}'

        train_data = get_dataset(args.data_dir, args.split_file, args.dataset_type, 'train', augment=True)
        train_ture_data = get_dataset(args.data_dir, args.split_file, args.dataset_type, 'train', augment=False)
        val_data = get_dataset(args.data_dir, args.split_file, args.dataset_type, 'val', augment=False)
        test_data = get_dataset(args.data_dir, args.split_file, args.dataset_type, 'test', augment=False)

        train_loader = DataLoader(dataset=train_data, batch_size=args.bs, shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=True)
        train_ture_loader = DataLoader(dataset=train_ture_data, batch_size=args.bs, shuffle=False,
                                       num_workers=4, pin_memory=True, drop_last=False)
        val_loader = DataLoader(dataset=val_data, batch_size=args.bs, shuffle=False,
                                num_workers=4, pin_memory=True, drop_last=False)
        test_loader = DataLoader(dataset=test_data, batch_size=args.bs, shuffle=False,
                                 num_workers=4, pin_memory=True, drop_last=False)

        print(f'train_length: {train_data.len}  val_length: {val_data.len}')
        return train_loader, train_ture_loader, val_loader, test_loader, save_dir


def forward_model(net, batch, dataset_type):
    """
    Forward pass through the model based on dataset type.

    Batch structures for each dataset_type:
    - imgonlyMR: (ID, img_MR_DCE, labels)                          -> 1 image
    - imgonlyMG: (ID, img_MG_CC, img_MG_MLO, labels)               -> 2 images
    - imgonlyUS: (ID, img_US_H, img_US_V, labels)                  -> 2 images
    - imgonly: (ID, img_MR_DCE, img_MG_CC, img_MG_MLO, img_US_H, img_US_V, labels) -> 5 images
    - imgonlyMRMG: (ID, img_MR_DCE, img_MG_CC, img_MG_MLO, labels) -> 3 images
    - imgonlyMRUS: (ID, img_MR_DCE, img_US_H, img_US_V, labels)    -> 3 images
    - imgonlyMGUS: (ID, img_MG_CC, img_MG_MLO, img_US_H, img_US_V, labels) -> 4 images
    - allimgclinical: (ID, 5 images, clinical_features, pathology_features, labels) -> 5 images + 2 features
    - onlyclinical: (ID, clinical_features, pathology_features, labels) -> 2 features
    """

    # imgonlyMR: 1 image (MR_DCE)
    if dataset_type == 'imgonlyMR':
        ID, img1, labels = batch
        img1 = img1.cuda().float()
        _, outputs = net(img1)

    # imgonlyMG: 2 images (MG_CC, MG_MLO)
    elif dataset_type == 'imgonlyMG':
        ID, img1, img2, labels = batch
        img1 = img1.cuda().float()
        img2 = img2.cuda().float()
        _, _, outputs = net(img1, img2)

    # imgonlyUS: 2 images (US_H, US_V)
    elif dataset_type == 'imgonlyUS':
        ID, img1, img2, labels = batch
        img1 = img1.cuda().float()
        img2 = img2.cuda().float()
        _, _, outputs = net(img1, img2)

    # imgonly: 5 images (MR_DCE, MG_CC, MG_MLO, US_H, US_V)
    elif dataset_type == 'imgonly':
        ID, img1, img2, img3, img4, img5, labels = batch
        img1 = img1.cuda().float()
        img2 = img2.cuda().float()
        img3 = img3.cuda().float()
        img4 = img4.cuda().float()
        img5 = img5.cuda().float()
        _, _, _, _, _, outputs = net(img1, img2, img3, img4, img5)

    # imgonlyMRMG: 3 images (MR_DCE, MG_CC, MG_MLO)
    elif dataset_type == 'imgonlyMRMG':
        ID, img1, img2, img3, labels = batch
        img1 = img1.cuda().float()
        img2 = img2.cuda().float()
        img3 = img3.cuda().float()
        _, _, _, outputs = net(img1, img2, img3)

    # imgonlyMRUS: 3 images (MR_DCE, US_H, US_V)
    elif dataset_type == 'imgonlyMRUS':
        ID, img1, img2, img3, labels = batch
        img1 = img1.cuda().float()
        img2 = img2.cuda().float()
        img3 = img3.cuda().float()
        _, _, _, outputs = net(img1, img2, img3)

    # imgonlyMGUS: 4 images (MG_CC, MG_MLO, US_H, US_V)
    elif dataset_type == 'imgonlyMGUS':
        ID, img1, img2, img3, img4, labels = batch
        img1 = img1.cuda().float()
        img2 = img2.cuda().float()
        img3 = img3.cuda().float()
        img4 = img4.cuda().float()
        _, _, _, _, outputs = net(img1, img2, img3, img4)

    # allimgclinical / allimgclinical_notdistance: 5 images + 2 clinical-pathology features
    elif dataset_type in ['allimgclinical', 'allimgclinical_notdistance']:
        ID, img_MR_DCE, img_MG_CC, img_MG_MLO, img_US_H, img_US_V, clinical_features, pathology_features, labels = batch
        img_MR_DCE = img_MR_DCE.cuda().float()
        img_MG_CC = img_MG_CC.cuda().float()
        img_MG_MLO = img_MG_MLO.cuda().float()
        img_US_H = img_US_H.cuda().float()
        img_US_V = img_US_V.cuda().float()
        clinical_features = clinical_features.cuda().float()
        pathology_features = pathology_features.cuda().float()
        _, _, _, _, _, _, _, outputs = net(img_MR_DCE, img_MG_CC, img_MG_MLO,
                                           img_US_H, img_US_V, clinical_features, pathology_features)

    # onlyclinical / onlyclinical_notdistance: 2 clinical-pathology features
    elif dataset_type in ['onlyclinical', 'onlyclinical_notdistance']:
        ID, clinical_features, pathology_features, labels = batch
        clinical_features = clinical_features.cuda().float()
        pathology_features = pathology_features.cuda().float()
        _, _, outputs = net(clinical_features, pathology_features)

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    return outputs, labels, ID


def evaluate_epoch(net, dataloader, loss_fc, num_class, dataset_type):
    """Evaluate model for one epoch"""
    net.eval()
    epoch_loss = []
    epoch_label = []
    epoch_pred_scores = []
    epoch_class_label = []
    epoch_pred_class = []
    epoch_ID = []

    with torch.no_grad():
        for batch in dataloader:
            outputs, labels, ID = forward_model(net, batch, dataset_type)

            labels = labels.cuda().long()
            labels_one_hot = torch.zeros((labels.size(0), num_class)).cuda().scatter_(
                1, labels.unsqueeze(1), 1).float().cpu()

            loss = loss_fc(outputs, labels)
            outputs = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, dim=1, keepdim=False).detach()

            epoch_pred_scores.append(outputs.detach().cpu())
            epoch_label.append(labels_one_hot)
            epoch_loss.append(loss.item())
            epoch_class_label.append(labels.cpu().numpy())
            epoch_pred_class.append(predicted.cpu().numpy())
            epoch_ID.append(ID)

    # Concatenate all batch results
    epoch_label = torch.cat(epoch_label, dim=0).numpy().astype(np.uint8)
    epoch_pred_scores = torch.cat(epoch_pred_scores, dim=0).numpy()
    epoch_class_label = np.concatenate(epoch_class_label)
    epoch_pred_class = np.concatenate(epoch_pred_class)
    epoch_ID = np.concatenate(epoch_ID)
    epoch_loss = np.mean(epoch_loss)

    # Calculate metrics
    auc = roc_auc_score(epoch_label, epoch_pred_scores)
    acc = accuracy_score(epoch_class_label, epoch_pred_class)

    return auc, acc, epoch_loss, epoch_class_label, epoch_pred_class, epoch_pred_scores, epoch_ID


def train_epoch(net, train_loader, optimizer, loss_fc, loss_threshold, num_class, epoch, args):
    """Train model for one epoch"""
    net.train()
    train_epoch_loss = []
    train_epoch_one_hot_label = []
    train_epoch_pred_scores = []
    train_epoch_class_label = []
    train_epoch_pred_class = []
    train_epoch_ID = []

    for i, batch in enumerate(train_loader):
        outputs, labels, ID = forward_model(net, batch, args.dataset_type)

        labels = labels.cuda().long()
        labels_one_hot = torch.zeros((labels.size(0), num_class)).cuda().scatter_(
            1, labels.unsqueeze(1), 1).float().cpu()

        optimizer.zero_grad()
        loss = loss_fc(outputs, labels)
        loss = (loss - loss_threshold).abs() + loss_threshold
        loss.backward()
        optimizer.step()

        outputs = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(outputs, dim=1, keepdim=False).detach()

        train_epoch_pred_scores.append(outputs.detach().cpu())
        train_epoch_one_hot_label.append(labels_one_hot)
        train_epoch_loss.append(loss.item())
        train_epoch_class_label.append(labels.cpu().numpy())
        train_epoch_pred_class.append(predicted.cpu().numpy())
        train_epoch_ID.append(ID)

        print(f'[{epoch + 1}/{args.epoch}, {i + 1}/{len(train_loader)}] train_loss: {loss.item():.3f}')

    # Concatenate all batch results
    train_epoch_one_hot_label = torch.cat(train_epoch_one_hot_label, dim=0).numpy().astype(np.uint8)
    train_epoch_pred_scores = torch.cat(train_epoch_pred_scores, dim=0).numpy()
    train_epoch_class_label = np.concatenate(train_epoch_class_label)
    train_epoch_pred_class = np.concatenate(train_epoch_pred_class)
    train_epoch_ID = np.concatenate(train_epoch_ID)
    train_epoch_loss = np.mean(train_epoch_loss)

    # Calculate metrics
    auc = roc_auc_score(train_epoch_one_hot_label, train_epoch_pred_scores)
    acc = accuracy_score(train_epoch_class_label, train_epoch_pred_class)

    return auc, acc, train_epoch_loss, train_epoch_class_label, train_epoch_pred_class, train_epoch_pred_scores, train_epoch_ID


def train(args):
    """Main training function"""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(args.seed)

    num_class = 2

    # Get data loaders
    train_loader, train_ture_loader, val_loader, test_loader, save_dir = get_data_loaders(args, is_external=False)

    # Create directories and tensorboard writers
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'log/train'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'log/train_ture'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'log/val'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'log/test'), exist_ok=True)

    train_writer = SummaryWriter(os.path.join(save_dir, 'log/train'), flush_secs=2)
    train_ture_writer = SummaryWriter(os.path.join(save_dir, 'log/train_ture'), flush_secs=2)
    val_writer = SummaryWriter(os.path.join(save_dir, 'log/val'), flush_secs=2)
    test_writer = SummaryWriter(os.path.join(save_dir, 'log/test'), flush_secs=2)

    print(f"Model save directory: {save_dir}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Split type: {args.split_type}")
    print(f"Model: {args.model}")

    # Initialize model
    net = get_model(args, num_class)

    # Load pretrained weights if specified
    if args.pretrained_path:
        org_params = torch.load(args.pretrained_path)
        model_dict = net.state_dict()
        need_org_params = {k: v for k, v in org_params.items()
                           if k in model_dict and k.split('.', 1)[0] != 'Trans'}
        model_dict.update(need_org_params)
        net.load_state_dict(model_dict)
        print(f"Loaded pretrained weights from {args.pretrained_path}")

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = MultiStepLR(optimizer, milestones=[int(0.9 * args.epoch)], gamma=0.1)
    loss_fc = nn.CrossEntropyLoss()

    best_AUC_val, best_AUC_test = 0, 0

    # Training loop
    for epoch in range(args.epoch):
        lr = optimizer.param_groups[0]['lr']

        # Train one epoch
        train_AUC, train_ACC, train_loss, train_class_label, train_pred_class, train_pred_scores, train_ids = train_epoch(
            net, train_loader, optimizer, loss_fc, args.loss_threshold, num_class, epoch, args)
        lr_scheduler.step()

        # Evaluate on validation, test, and train_ture sets
        val_AUC, val_ACC, val_loss, val_true_labels, val_pred_labels, val_pred_scores, val_ids = evaluate_epoch(
            net, val_loader, loss_fc, num_class, args.dataset_type)
        test_AUC, test_ACC, test_loss, test_true_labels, test_pred_labels, test_pred_scores, test_ids = evaluate_epoch(
            net, test_loader, loss_fc, num_class, args.dataset_type)
        train_ture_AUC, train_ture_ACC, train_ture_loss, train_ture_true_labels, train_ture_pred_labels, train_ture_pred_scores, train_ture_ids = evaluate_epoch(
            net, train_ture_loader, loss_fc, num_class, args.dataset_type)

        print(f'[{epoch}/{args.epoch}] train_loss: {train_loss:.3f} train_AUC: {train_AUC:.3f} '
              f'train_ture_AUC: {train_ture_AUC:.3f} val_AUC: {val_AUC:.3f} test_AUC: {test_AUC:.3f}')

        # Save best models
        if val_AUC > best_AUC_val:
            best_AUC_val = val_AUC
            torch.save(net.state_dict(), os.path.join(save_dir, 'best_AUC_val.pth'))
        if test_AUC > best_AUC_test:
            best_AUC_test = test_AUC
            torch.save(net.state_dict(), os.path.join(save_dir, 'best_AUC_test.pth'))

        # Log to tensorboard
        train_writer.add_scalar('lr', lr, epoch)
        train_writer.add_scalar('loss', train_loss, epoch)
        train_writer.add_scalar('AUC', train_AUC, epoch)
        train_writer.add_scalar('ACC', train_ACC, epoch)

        val_writer.add_scalar('loss', val_loss, epoch)
        val_writer.add_scalar('AUC', val_AUC, epoch)
        val_writer.add_scalar('ACC', val_ACC, epoch)
        val_writer.add_scalar('best_AUC_val', best_AUC_val, epoch)

        test_writer.add_scalar('loss', test_loss, epoch)
        test_writer.add_scalar('AUC', test_AUC, epoch)
        test_writer.add_scalar('ACC', test_ACC, epoch)
        test_writer.add_scalar('best_AUC_test', best_AUC_test, epoch)

        train_ture_writer.add_scalar('AUC', train_ture_AUC, epoch)
        train_ture_writer.add_scalar('ACC', train_ture_ACC, epoch)

        # Save predictions and labels for each epoch
        # Training set (with augmentation)
        np.save(os.path.join(save_dir, f'log/train/truelabel_{epoch}.npy'), train_class_label)
        np.save(os.path.join(save_dir, f'log/train/predlabel_{epoch}.npy'), train_pred_class)
        np.save(os.path.join(save_dir, f'log/train/predscores_{epoch}.npy'), train_pred_scores)
        np.save(os.path.join(save_dir, f'log/train/ID_{epoch}.npy'), train_ids)

        # Training set (without augmentation)
        np.save(os.path.join(save_dir, f'log/train_ture/truelabel_{epoch}.npy'), train_ture_true_labels)
        np.save(os.path.join(save_dir, f'log/train_ture/predlabel_{epoch}.npy'), train_ture_pred_labels)
        np.save(os.path.join(save_dir, f'log/train_ture/predscores_{epoch}.npy'), train_ture_pred_scores)
        np.save(os.path.join(save_dir, f'log/train_ture/ID_{epoch}.npy'), train_ture_ids)

        # Validation set
        np.save(os.path.join(save_dir, f'log/val/truelabel_{epoch}.npy'), val_true_labels)
        np.save(os.path.join(save_dir, f'log/val/predlabel_{epoch}.npy'), val_pred_labels)
        np.save(os.path.join(save_dir, f'log/val/predscores_{epoch}.npy'), val_pred_scores)
        np.save(os.path.join(save_dir, f'log/val/ID_{epoch}.npy'), val_ids)

        # Test set
        np.save(os.path.join(save_dir, f'log/test/truelabel_{epoch}.npy'), test_true_labels)
        np.save(os.path.join(save_dir, f'log/test/predlabel_{epoch}.npy'), test_pred_labels)
        np.save(os.path.join(save_dir, f'log/test/predscores_{epoch}.npy'), test_pred_scores)
        np.save(os.path.join(save_dir, f'log/test/ID_{epoch}.npy'), test_ids)

    # Cleanup
    train_writer.close()
    val_writer.close()
    test_writer.close()
    train_ture_writer.close()
    print(f'Model saved: {save_dir}')


def test(args, is_external=False):
    """Main testing function"""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(args.seed)

    num_class = 2

    # Get data loaders
    if is_external:
        _, _, test_loader, save_dir = get_data_loaders(args, is_external=True)
    else:
        _, _, _, test_loader, save_dir = get_data_loaders(args, is_external=False)

    # Create directory and tensorboard writer
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'log/test'), exist_ok=True)
    writer = SummaryWriter(os.path.join(save_dir, 'log/test'), flush_secs=2)

    print(f"Test save directory: {save_dir}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Split type: {args.split_type}")

    # Load model
    net = get_model(args, num_class)
    model_path = os.path.join(
        f'./trained_models/{args.name}_{args.split_type}/bs4_epoch{args.epoch}_seed{args.seed}_{args.date}_drop{args.dropout}',
        'best_AUC_test.pth')
    net.load_state_dict(torch.load(model_path))
    print(f"Loaded model from {model_path}")

    loss_fc = nn.CrossEntropyLoss()

    # Evaluate
    test_AUC, test_ACC, test_loss, true_labels, pred_labels, pred_scores, ids = evaluate_epoch(
        net, test_loader, loss_fc, num_class, args.dataset_type)

    print(f'Test Results - AUC: {test_AUC:.4f}, ACC: {test_ACC:.4f}, Loss: {test_loss:.4f}')

    # Save results
    writer.add_scalar('loss', test_loss, 0)
    writer.add_scalar('AUC', test_AUC, 0)
    writer.add_scalar('ACC', test_ACC, 0)

    np.save(os.path.join(save_dir, 'log/test/truelabel_0.npy'), true_labels)
    np.save(os.path.join(save_dir, 'log/test/predlabel_0.npy'), pred_labels)
    np.save(os.path.join(save_dir, 'log/test/predscores_0.npy'), pred_scores)
    np.save(os.path.join(save_dir, 'log/test/ID_0.npy'), ids)

    writer.close()
    print(f'Results saved: {save_dir}')


if __name__ == '__main__':
    args = get_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args, is_external=False)
    elif args.mode == 'test_external':
        test(args, is_external=True)