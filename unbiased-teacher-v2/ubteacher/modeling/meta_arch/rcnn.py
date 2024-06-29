# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads


from .gdl import decouple_layer, AffineLayer
#from .gdl import decouple_layer
import torch
from typing import Dict, List, Optional
from detectron2.structures import Instances


@META_ARCH_REGISTRY.register()

class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):

    def __init__(self, cfg):

        backbone = build_backbone(cfg)
        _SHAPE_ = backbone.output_shape()
        proposal_generator = build_proposal_generator(cfg, _SHAPE_)
        roi_heads = build_roi_heads(cfg, _SHAPE_)
        super().__init__(backbone=backbone, proposal_generator=proposal_generator, 
                         roi_heads=roi_heads, pixel_mean=cfg.MODEL.PIXEL_MEAN, pixel_std=cfg.MODEL.PIXEL_STD)
        self._SHAPE_=_SHAPE_
        self.cfg=cfg
        #Creanse os modulos GDL
        if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
            self.affine_rpn = AffineLayer(num_channels=_SHAPE_['p2'].channels, bias=True)
        if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
            self.affine_rcnn = AffineLayer(num_channels=_SHAPE_['p2'].channels, bias=True)
            
        #Engadense as constantes necesarias para facer a asignacion SimOTA
        self.roi_heads.simota_assignment=self.cfg.MODEL.SIMOTA_ASSIGNMENT
        if self.cfg.MODEL.SIMOTA_ASSIGNMENT:
            self.roi_heads.simota_coef_reg=self.cfg.MODEL.SIMOTA_COEF_REG
            self.roi_heads.simota_radius=self.cfg.MODEL.SIMOTA_RADIUS
            self.roi_heads.simota_q_dynamick=self.cfg.MODEL.SIMOTA_Q_DYNAMICK
        

    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        #Engadimos gdl antes de rpn y roi heads

        if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
            features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}

        else:
            features_de_rpn=features
        
        if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
            features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}

        else:
            features_de_rcnn=features

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features_de_rpn, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features_de_rcnn, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features_de_rpn, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features_de_rcnn,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "unsup_data_train":  #

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features_de_rpn, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features_de_rcnn, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None
        
    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        #GDL antes de rpn e roiheads
        if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
            features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}

        else:
            features_de_rpn=features
        
        if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
            features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}

        else:
            features_de_rcnn=features

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features_de_rpn, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features_de_rcnn, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        return results
