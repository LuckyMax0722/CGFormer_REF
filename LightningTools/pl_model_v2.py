import os
import torch
import numpy as np
import pytorch_lightning as pl
from .basemodel import LightningBaseModel
from .metric import SSCMetrics
from mmdet3d.models import build_model
from .utils import get_inv_map
from mmcv.runner.checkpoint import load_checkpoint


class pl_model(LightningBaseModel):
    def __init__(
        self,
        config):
        super(pl_model, self).__init__(config)

        model_config = config['model']
        self.model = build_model(model_config)
        
        if 'load_from' in config:
            load_checkpoint(self.model, 
                        '/u/home/caoh/projects/MA_Jiachen/CGFormer/ckpts/CGFormer-Efficient-Swin-SemanticKITTI.ckpt', 
                        map_location='cpu', 
                        strict=False,
                        revise_keys=[(r'model.', '')])
        
        self.num_class = config['num_class']
        self.class_names = config['class_names']

        self.train_metrics = SSCMetrics(config['num_class'])
        self.val_metrics = SSCMetrics(config['num_class'])
        self.test_metrics = SSCMetrics(config['num_class'])
        self.save_path = config['save_path']
        self.test_mapping = config['test_mapping']
        self.pretrain = config['pretrain']
        
        self.freeze_model_layers()
    
    def forward(self, data_dict):
        return self.model(data_dict)
    
    def freeze_model_layers(self):
        self.model.img_backbone.requires_grad_(False)
        self.model.img_backbone.eval()
        
        self.model.img_neck.requires_grad_(False)
        self.model.img_neck.eval()
            
        self.model.depth_net.requires_grad_(False)
        self.model.depth_net.eval()
        
        self.model.img_view_transformer.requires_grad_(False)
        self.model.img_view_transformer.eval()
        
        self.model.proposal_layer.requires_grad_(False)
        self.model.proposal_layer.eval()
        
        self.model.VoxFormer_head.requires_grad_(False)
        self.model.VoxFormer_head.eval()
         
        self.model.occ_encoder_backbone.requires_grad_(False)
        self.model.occ_encoder_backbone.eval()
        
        self.model.pts_bbox_head.requires_grad_(False)
        self.model.pts_bbox_head.eval()
            
    def training_step(self, batch, batch_idx):
        output_dict = self.forward(batch)
        loss_dict = output_dict['losses']
        loss = 0
        for key, value in loss_dict.items():
            self.log(
                "train/"+key,
                value.detach(),
                on_epoch=True,
                sync_dist=True)
            loss += value
            
        self.log("train/loss",
            loss.detach(),
            on_epoch=True,
            sync_dist=True)
        
        if not self.pretrain:
            pred = output_dict['pred'].detach().cpu().numpy()
            gt_occ = output_dict['gt_occ'].detach().cpu().numpy()
            
            self.train_metrics.add_batch(pred, gt_occ)

        return loss
    
    def validation_step(self, batch, batch_idx):
        
        output_dict = self.forward(batch)
        
        if not self.pretrain:
            pred = output_dict['pred'].detach().cpu().numpy()
            gt_occ = output_dict['gt_occ'].detach().cpu().numpy()

            self.val_metrics.add_batch(pred, gt_occ)
    
    def validation_epoch_end(self, outputs):
        metric_list = [("train", self.train_metrics), ("val", self.val_metrics)]
        # metric_list = [("val", self.val_metrics)]
        
        metrics_list = metric_list
        
        for prefix, metric in metrics_list:
            stats = metric.get_stats()

            if prefix == 'val':
                for name, iou in zip(self.class_names, stats['iou_ssc']):
                    self.log("{}/{}/IoU".format(prefix, name), torch.tensor(iou, dtype=torch.float32), sync_dist=True)
                
            self.log("{}/mIoU".format(prefix), torch.tensor(stats["iou_ssc_mean"], dtype=torch.float32), sync_dist=True)
            self.log("{}/IoU".format(prefix), torch.tensor(stats["iou"], dtype=torch.float32), sync_dist=True)
            self.log("{}/Precision".format(prefix), torch.tensor(stats["precision"], dtype=torch.float32), sync_dist=True)
            self.log("{}/Recall".format(prefix), torch.tensor(stats["recall"], dtype=torch.float32), sync_dist=True)
            
            metric.reset()
        
    def test_step(self, batch, batch_idx):
        output_dict = self.forward(batch)

        pred = output_dict['pred'].detach().cpu().numpy()
        gt_occ = output_dict['gt_occ']
        if gt_occ is not None:
            gt_occ = gt_occ.detach().cpu().numpy()
        else:
            gt_occ = None
            
        if self.save_path is not None:
            if self.test_mapping:
                inv_map = get_inv_map()
                output_voxels = inv_map[pred].astype(np.uint16)
            else:
                output_voxels = pred.astype(np.uint16)
            sequence_id = batch['img_metas']['sequence'][0]
            frame_id = batch['img_metas']['frame_id'][0]
            save_folder = "{}/sequences/{}/predictions".format(self.save_path, sequence_id)
            save_file = os.path.join(save_folder, "{}.label".format(frame_id))
            os.makedirs(save_folder, exist_ok=True)
            with open(save_file, 'wb') as f:
                output_voxels.tofile(f)
                print('\n save to {}'.format(save_file))
            
        if gt_occ is not None:
            self.test_metrics.add_batch(pred, gt_occ)
    
    def test_epoch_end(self, outputs):
        metric_list = [("test", self.test_metrics)]
        # metric_list = [("val", self.val_metrics)]
        metrics_list = metric_list
        for prefix, metric in metrics_list:
            stats = metric.get_stats()

            for name, iou in zip(self.class_names, stats['iou_ssc']):
                print(name + ":", iou)

            self.log("{}/mIoU".format(prefix), torch.tensor(stats["iou_ssc_mean"], dtype=torch.float32), sync_dist=True)
            self.log("{}/IoU".format(prefix), torch.tensor(stats["iou"], dtype=torch.float32), sync_dist=True)
            self.log("{}/Precision".format(prefix), torch.tensor(stats["precision"], dtype=torch.float32), sync_dist=True)
            self.log("{}/Recall".format(prefix), torch.tensor(stats["recall"], dtype=torch.float32), sync_dist=True)
            metric.reset()
    