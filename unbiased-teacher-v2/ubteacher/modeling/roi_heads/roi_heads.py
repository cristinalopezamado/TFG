# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from detectron2.layers import ShapeSpec
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals,
)
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from ubteacher.modeling.roi_heads.fast_rcnn import (
    FastRCNNCrossEntropyBoundaryVarOutputLayers,
    FastRCNNFocaltLossBoundaryVarOutputLayers,
    FastRCNNFocaltLossOutputLayers,
)
import detectron2.utils.comm as comm
from detectron2.modeling.poolers import assign_boxes_to_levels



@ROI_HEADS_REGISTRY.register()
class StandardROIHeadsPseudoLab(StandardROIHeads):
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels, height=pooler_resolution, width=pooler_resolution
            ),
        )
        if cfg.MODEL.ROI_HEADS.LOSS == "CrossEntropy":
            box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        elif cfg.MODEL.ROI_HEADS.LOSS == "FocalLoss":
            box_predictor = FastRCNNFocaltLossOutputLayers(cfg, box_head.output_shape)
        elif cfg.MODEL.ROI_HEADS.LOSS == "FocalLoss_BoundaryVar":
            box_predictor = FastRCNNFocaltLossBoundaryVarOutputLayers(
                cfg, box_head.output_shape
            )
        elif cfg.MODEL.ROI_HEADS.LOSS == "CrossEntropy_BoundaryVar":
            box_predictor = FastRCNNCrossEntropyBoundaryVarOutputLayers(
                cfg, box_head.output_shape
            )

        else:
            raise ValueError("Unknown ROI head loss.")

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        compute_loss=True,
        branch="",
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:

        del images
        if self.training and compute_loss:  # apply if training loss
            assert targets
            # 1000 --> 512
            if targets[0].has("scores"):  # has confidence; then weight loss
                proposals = self.label_and_sample_proposals_pseudo(
                    proposals, targets, branch=branch
                )
            else:
                proposals = self.label_and_sample_proposals(
                    proposals, targets, branch=branch
                )

        #del targets

        if self.training and compute_loss:
            losses, _ = self._forward_box(features, proposals, compute_loss, branch,targets)
            return proposals, losses
        else:
            pred_instances, predictions = self._forward_box(
                features, proposals, compute_loss, branch
            )

            return pred_instances, predictions
    def get_geometry_constraint(self,gt_boxes,prop_boxes,strides):
        #Coordenada (x,y) do centro de cada caixa
        gt_centers=torch.stack(((gt_boxes[:,2]+gt_boxes[:,0])/2,(gt_boxes[:,3]+gt_boxes[:,1])/2),dim=1)
        prop_centers=torch.stack(((prop_boxes[:,2]+prop_boxes[:,0])/2,(prop_boxes[:,3]+prop_boxes[:,1])/2),dim=1)
           
        r=self.simota_radius*strides
        #Caixa co mesmo centro e radio r
        gt_radio_l=gt_centers[:,0]-r.unsqueeze(1)
        gt_radio_r=gt_centers[:,0]+r.unsqueeze(1)
        gt_radio_b=gt_centers[:,1]-r.unsqueeze(1)
        gt_radio_t=gt_centers[:,1]+r.unsqueeze(1)
        
        c_l=(prop_centers[:,0].unsqueeze(1)-gt_radio_l).transpose(0,1)
        c_r=(gt_radio_r-prop_centers[:,0].unsqueeze(1)).transpose(0,1)
        c_b=(prop_centers[:,1].unsqueeze(1)-gt_radio_b).transpose(0,1)
        c_t=(gt_radio_t-prop_centers[:,1].unsqueeze(1)).transpose(0,1)
        
        center_deltas = torch.stack([c_l, c_b, c_r, c_t], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        anchor_filter = is_in_centers.sum(dim=0) > 0
        geometry_relation = is_in_centers[:, anchor_filter]

        return anchor_filter, geometry_relation

    #Funcion simota_matching de yolox
    def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k = min(self.simota_q_dynamick, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        if cost.shape[1]==0:
            hola=1
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        # deal with the case that one anchor matches multiple ground-truths
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[:, multiple_match_mask] *= 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1
        fg_mask_inboxes = anchor_matching_gt > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
    
    
    def asignacionSIMOTA(self,predictions,proposals,targets):
        
        result_proposals=[]
        scores, proposal_deltas, proposal_deltas_std = predictions
        #Obtemos os strides de cada caixa
        assign_levels=assign_boxes_to_levels([p.proposal_boxes for p in proposals],self.box_pooler.min_level,self.box_pooler.max_level,self.box_pooler.canonical_box_size,self.box_pooler.canonical_level)
        strides=torch.pow(2,assign_levels+2)
        num_predictions_per_image=proposals[0].gt_classes.shape[0]
        inicioPredictions=0
        
        mask_usados=torch.ones(scores.shape[0], dtype=torch.bool,device=scores.device)

        #Recorro as imaxes
        for index,(image_target,image_proposal) in enumerate(zip(targets,proposals)):
            
            #Scores e strides da imaxe considerada   
            predicted_classes_total=scores[inicioPredictions:inicioPredictions+num_predictions_per_image]
            strides_total=strides[inicioPredictions:inicioPredictions+num_predictions_per_image]

            #Indices que non teñen asignado fondo
            mask_nofondo= image_proposal.gt_classes!=80

            #So aplicamos simota as propostas que tiñan asignado obxecto
            predicted_classes_init=predicted_classes_total[mask_nofondo]
            predicted_boxes_init=image_proposal.proposal_boxes[mask_nofondo]
            strides_init=strides_total[mask_nofondo]
            
            #So consideraremos as asignacions proposta-gt se a proposta esta suficientemente preto do gt              
            fg_mask, geometry_relation=self.get_geometry_constraint(image_target.gt_boxes.tensor,predicted_boxes_init.tensor,strides_init)

            gt_classes=image_target.gt_classes                
            predicted_classes=predicted_classes_init[fg_mask]
            predicted_boxes=predicted_boxes_init[fg_mask]
            
            #Calculo a matriz de custos
            if predicted_classes.numel()!=0:
                clas_cost_matrix=[]
                #Recorro os gt desa imaxe e calculo os custos (comparando con cada proposta)
                for gt_c in gt_classes:
                    lista_gt=torch.full((len(predicted_classes),),gt_c,device=predicted_classes.device)
                    #Coste de clasificacion de asignar cada proposta a ese gt (focal loss)
                    ce=torch.nn.functional.cross_entropy(predicted_classes,lista_gt,reduction='none')
                    p = torch.exp(-ce)
                    clas_cost=(1 - p) ** 1.5 * ce
                    
                    clas_cost_matrix.append(clas_cost)
                clas_cost_matrix=torch.vstack(clas_cost_matrix)
                
                #Coste de regresion
                iou_matrix=pairwise_iou(image_target.gt_boxes, predicted_boxes)
                reg_cost_matrix=-torch.log(iou_matrix+1e-8)
                #Balancing coefficient lambda
                bal_coef=self.simota_coef_reg
                #Matriz de coste: filas gt, columnas proposals
                cost_matrix=clas_cost_matrix+bal_coef*reg_cost_matrix+1e6*(~geometry_relation)
                #Asignacion simota
                (
                    num_fg,
                    gt_matched_classes,
                    pred_ious_this_matching,
                    matched_gt_inds,
                )=self.simota_matching(cost_matrix, iou_matrix, gt_classes, len(gt_classes), fg_mask)

                
            index_fg_mask=0
            index_gt=0
            #Campos das propostas resultado
            result_gt_boxes=[]
            result_gt_classes=[]
            result_objectness_logits=[]
            result_proposal_boxes=[]
            result_gt_confid=[]
            result_gt_loc_std=[]
            for i in range(len(proposals[index].proposal_boxes)):
                if mask_nofondo[i]:
                    if fg_mask[index_fg_mask]:
                        
                        #Se ten unha asignación de gt en simota engadese ao resultado
                        j=matched_gt_inds[index_gt]
                        gt_box=targets[index].gt_boxes[j.item()].tensor[0]
                        gt_clas=targets[index].gt_classes[j.item()]
                        prop_box=proposals[index].proposal_boxes[i].tensor[0]
                        obj_log=proposals[index].objectness_logits[i]
                        if proposals[index].has('gt_confid'):
                            gt_confid=proposals[index].gt_confid[i]
                        if proposals[index].has('gt_loc_std'):
                            gt_loc_std=proposals[index].gt_loc_std[i]
                        
                        result_gt_boxes.append(gt_box)
                        result_gt_classes.append(gt_clas)
                        result_objectness_logits.append(obj_log)
                        result_proposal_boxes.append(prop_box)
                        if proposals[index].has('gt_confid'):
                            result_gt_confid.append(gt_confid)
                        if proposals[index].has('gt_loc_std'):
                            result_gt_loc_std.append(gt_loc_std)
                        
                        index_gt+=1
                        index_fg_mask+=1
                    else:
                        #se non foi asignado en simota ignorase
                        index_fg_mask+=1
                        mask_usados[index*num_predictions_per_image+i]=False
                else:
                    #se era fondo pola rpn deixase como estaba: segue sendo fondo
                    gt_clas=proposals[index].gt_classes[i]
                    gt_box=proposals[index].gt_boxes[i].tensor[0]
                    prop_box=proposals[index].proposal_boxes[i].tensor[0]
                    obj_log=proposals[index].objectness_logits[i]
                    if proposals[index].has('gt_confid'):
                        gt_confid=proposals[index].gt_confid[i]
                    if proposals[index].has('gt_loc_std'):
                        gt_loc_std=proposals[index].gt_loc_std[i]
                    
                    result_gt_boxes.append(gt_box)
                    result_gt_classes.append(gt_clas)
                    result_objectness_logits.append(obj_log)
                    result_proposal_boxes.append(prop_box)
                    if proposals[index].has('gt_confid'):
                        result_gt_confid.append(gt_confid)
                    if proposals[index].has('gt_loc_std'):
                        result_gt_loc_std.append(gt_loc_std)
            
            #Campos finais para esa imaxe                                        
            result_gt_boxes=Boxes(torch.stack(result_gt_boxes))
            result_gt_classes=torch.stack(result_gt_classes)
            result_objectness_logits=torch.stack(result_objectness_logits)
            
            if proposals[index].has('gt_confid'):
                result_gt_confid=torch.stack(result_gt_confid)
            if proposals[index].has('gt_loc_std'):
                result_gt_loc_std=torch.stack(result_gt_loc_std)
            
            result_proposal_boxes=Boxes(torch.stack(result_proposal_boxes))
            
            #Instancia final para esa imaxe
            result=Instances(proposals[index].image_size)
            result.proposal_boxes=result_proposal_boxes
            result.objectness_logits=result_objectness_logits
            result.gt_classes=result_gt_classes
            result.gt_boxes=result_gt_boxes
            if proposals[index].has('gt_confid'):
                result.gt_confid=result_gt_confid
            if proposals[index].has('gt_loc_std'):
                result.gt_loc_std=result_gt_loc_std
            result_proposals.append(result)
            
            inicioPredictions+=num_predictions_per_image

        #Devolvese scores, proposal_deltas e proposal_deltas_std das instancias que se van usar
        result_scores=scores[mask_usados]
        result_proposal_deltas=proposal_deltas[mask_usados]
        result_proposal_deltas_std=proposal_deltas_std[mask_usados]
        result_predictions=(result_scores,result_proposal_deltas,result_proposal_deltas_std)

        return result_proposals,result_predictions

    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        compute_loss: bool = True,
        branch: str = "",
        targets: Dict[str, torch.Tensor]={},
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training and compute_loss:  # apply if training loss or val loss
            if self.simota_assignment:
                proposals,predictions=self.asignacionSIMOTA(predictions,proposals,targets)
            losses = self.box_predictor.losses(predictions, proposals, branch)

            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                        proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses, predictions
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)

            return pred_instances, predictions

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances], branch: str = ""
    ) -> List[Instances]:
        gt_boxes = [x.gt_boxes for x in targets]

        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        storage = get_event_storage()
        storage.put_scalar(
            "roi_head/num_target_fg_samples_" + branch, np.mean(num_fg_samples)
        )
        storage.put_scalar(
            "roi_head/num_target_bg_samples_" + branch, np.mean(num_bg_samples)
        )

        return proposals_with_gt
    
    #Modificase matched_labels para marcar con -1 as propostas que se ignoran
    #Targets debe conter polo menos gt_ignore e gt_classes
    def ignore_proposals(self,matched_idxs,matched_labels,gt_ignore):
        for i in range(len(matched_idxs)):
            index=matched_idxs[i]
            #Se se asignou unha clase a unha gt que se ignora, marcase como ignore
            if gt_ignore[index] and matched_labels[i]==1:
                matched_labels[i]=-1
        return matched_labels

    @torch.no_grad()
    def label_and_sample_proposals_pseudo(
        self, proposals: List[Instances], targets: List[Instances], branch: str = ""
    ) -> List[Instances]:
        gt_boxes = [x.gt_boxes for x in targets]
        gt_confids = [x.scores for x in targets]
        if targets[0].has("pred_boxes_std"):
            gt_loc_std = [x.pred_boxes_std for x in targets]
        else:
            gt_loc_std = [None for x in targets]

        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []

        for (
            proposals_per_image,
            targets_per_image,
            confids_per_image,
            loc_std_per_image,
        ) in zip(proposals, targets, gt_confids, gt_loc_std):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            #Marcase con -1 aquelas propostas que se ignoran para a funcion de coste
            if targets_per_image.has("ignore") and targets_per_image.ignore.numel()!=0:
                matched_labels=self.ignore_proposals(matched_idxs,matched_labels,targets_per_image.ignore)

            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
                proposals_per_image.set("gt_confid", confids_per_image[sampled_targets])
                if loc_std_per_image is not None:
                    proposals_per_image.set(
                        "gt_loc_std", loc_std_per_image[sampled_targets, :]
                    )

            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes
                proposals_per_image.set("gt_confid", torch.zeros_like(sampled_idxs))
                if loc_std_per_image is not None:
                    proposals_per_image.set(
                        "gt_loc_std",
                        targets_per_image.gt_boxes.tensor.new_zeros(
                            (len(sampled_idxs), 4)
                        ),
                    )

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        storage = get_event_storage()
        storage.put_scalar(
            "roi_head/num_target_fg_samples_" + branch, np.mean(num_fg_samples)
        )
        storage.put_scalar(
            "roi_head/num_target_bg_samples_" + branch, np.mean(num_bg_samples)
        )

        return proposals_with_gt
