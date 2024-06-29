# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# for ema scheduler
import logging
import os
import time
from collections import OrderedDict

import detectron2.utils.comm as comm
import numpy as np
import torch
import random
from detectron2.engine import DefaultTrainer, hooks, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluator,
    print_csv_format,
    verify_results,
)
from detectron2.structures import Boxes
from detectron2.structures.instances import Instances
from detectron2.utils.events import EventStorage
from fvcore.nn.precise_bn import get_bn_modules
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel
import torchvision.transforms.functional
from ubteacher.checkpoint.detection_checkpoint import DetectionTSCheckpointer

from ubteacher.data.build import (
    build_detection_semisup_train_loader_two_crops,
    build_detection_test_loader, build_detection_pcb_loader
)
from ubteacher.data.dataset_mapper import DatasetMapperTwoCropSeparate, DatasetMapperPCB
from ubteacher.evaluation.evaluator import inference_on_dataset
from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from ubteacher.modeling.pseudo_generator import PseudoGenerator
from ubteacher.solver.build import build_lr_scheduler

from ubteacher.modeling.calibration_layer import PrototypicalCalibrationBlock
import cv2
import torchvision
import itertools
import matplotlib.pyplot as plt

# Unbiased Teacher Trainer for FCOS
class UBTeacherTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # create an student model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        # create an teacher model
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher
        self.model_teacher.eval()

        data_loader = self.build_train_loader(cfg)

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # Ensemble teacher and student model is for model saving and loading
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.pseudo_generator = PseudoGenerator(cfg)

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        if cfg.TEST.EVALUATOR == "COCOeval":
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        # elif cfg.TEST.EVALUATOR == "COCOTIDEeval":
        #     return COCOTIDEEvaluator(dataset_name, cfg, True, output_folder)
        else:
            return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()

                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step_full_semisup()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    # =====================================================
    # ================== Pseduo-labeling ==================
    # =====================================================
    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label, labeltype=""):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            if labeltype == "class":
                unlabel_datum["instances_class"] = lab_inst
            elif labeltype == "reg":
                unlabel_datum["instances_reg"] = lab_inst
            else:
                unlabel_datum["instances"] = lab_inst
        return unlabled_data

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
        data_time = time.perf_counter() - start
        # burn-in stage (supervised training with labeled data)
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:
            # input both strong and weak supervised data into model
            label_data_q.extend(label_data_k)
            if self.cfg.SOLVER.AMP.ENABLED:
                with autocast():
                    record_dict = self.model(label_data_q, branch="labeled")
            else:
                record_dict = self.model(label_data_q, branch="labeled")

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss" and key[-3:] != "val":
                    loss_dict[key] = record_dict[key]

            if self.cfg.SOLVER.AMP.ENABLED:
                with autocast():
                    losses = sum(loss_dict.values())
            else:
                losses = sum(loss_dict.values())

        else:
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                self._update_teacher_model(keep_rate=0.00)
                ema_keep_rate = self.cfg.SEMISUPNET.EMA_KEEP_RATE

            elif (
                self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
            ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:

                ema_keep_rate = self.cfg.SEMISUPNET.EMA_KEEP_RATE
                self._update_teacher_model(keep_rate=ema_keep_rate)

            record_dict = {}
            record_dict["ema_rate_1000x"] = ema_keep_rate * 1000
            # generate the pseudo-label using teacher model
            # note that we do not convert to eval mode, as 1) there is no gradient computed in
            # teacher model and 2) batch norm layers are not updated as well

            # produce raw prediction from teacher and predicted box after NMS (NMS_CRITERIA_TRAIN)
            with torch.no_grad():
                pred_teacher, raw_pred_teacher = self.model_teacher(
                    unlabel_data_k,
                    output_raw=True,
                    nms_method=self.cfg.MODEL.FCOS.NMS_CRITERIA_TRAIN,
                    branch="teacher_weak",
                )

            # use the above raw teacher prediction and perform another NMS (NMS_CRITERIA_REG_TRAIN)
            pred_teacher_loc = self.pseudo_generator.nms_from_dense(
                raw_pred_teacher, self.cfg.MODEL.FCOS.NMS_CRITERIA_REG_TRAIN
            )

            # set up threshold for pseudo-labeling
            ## pseudo-labeling for classification pseudo-labels
            if self.cfg.SEMISUPNET.PSEUDO_BBOX_SAMPLE == "thresholding":
                cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD
            elif self.cfg.SEMISUPNET.PSEUDO_BBOX_SAMPLE == "thresholding_cls_ctr":
                cur_threshold = (
                    self.cfg.SEMISUPNET.BBOX_THRESHOLD,
                    self.cfg.SEMISUPNET.BBOX_CTR_THRESHOLD,
                )
            else:
                raise ValueError

            ## pseudo-labeling for regression pseudo-labels
            if self.cfg.SEMISUPNET.PSEUDO_BBOX_SAMPLE_REG == "thresholding":
                cur_threshold_reg = self.cfg.SEMISUPNET.BBOX_THRESHOLD_REG
            elif self.cfg.SEMISUPNET.PSEUDO_BBOX_SAMPLE_REG == "thresholding_cls_ctr":
                cur_threshold_reg = (
                    self.cfg.SEMISUPNET.BBOX_THRESHOLD_REG,
                    self.cfg.SEMISUPNET.BBOX_CTR_THRESHOLD_REG,
                )
            else:
                raise ValueError

            # produce pseudo-labels
            joint_proposal_dict = {}

            # classification
            (
                pesudo_proposals_roih_unsup_k,
                _,
            ) = self.pseudo_generator.process_pseudo_label(
                pred_teacher,
                cur_threshold,
                "roih",
                self.cfg.SEMISUPNET.PSEUDO_BBOX_SAMPLE,
            )
            joint_proposal_dict["proposals_pseudo_cls"] = pesudo_proposals_roih_unsup_k

            # regression
            (
                pesudo_proposals_roih_unsup_k_reg,
                _,
            ) = self.pseudo_generator.process_pseudo_label(
                pred_teacher_loc,
                cur_threshold_reg,
                "roih",
                self.cfg.SEMISUPNET.PSEUDO_BBOX_SAMPLE_REG,
            )
            joint_proposal_dict[
                "proposals_pseudo_reg"
            ] = pesudo_proposals_roih_unsup_k_reg

            #  remove ground-truth labels from unlabeled data
            unlabel_data_q = self.remove_label(unlabel_data_q)
            unlabel_data_k = self.remove_label(unlabel_data_k)

            #  add pseudo-label to unlabeled data
            unlabel_data_q = self.add_label(
                unlabel_data_q, joint_proposal_dict["proposals_pseudo_cls"], "class"
            )
            unlabel_data_k = self.add_label(
                unlabel_data_k, joint_proposal_dict["proposals_pseudo_cls"], "class"
            )

            unlabel_data_q = self.add_label(
                unlabel_data_q, joint_proposal_dict["proposals_pseudo_reg"], "reg"
            )
            unlabel_data_k = self.add_label(
                unlabel_data_k, joint_proposal_dict["proposals_pseudo_reg"], "reg"
            )

            all_label_data = label_data_q + label_data_k
            all_unlabel_data = unlabel_data_q

            if self.cfg.SOLVER.AMP.ENABLED:
                with autocast():
                    record_all_label_data = self.model(all_label_data, branch="labeled")
            else:
                record_all_label_data = self.model(all_label_data, branch="labeled")
            record_dict.update(record_all_label_data)

            # unlabeled data pseudo-labeling
            for unlabel_data in all_unlabel_data:
                assert (
                    len(unlabel_data) != 0
                ), "unlabeled data must have at least one pseudo-box"

            if self.cfg.SOLVER.AMP.ENABLED:
                with autocast():
                    (
                        record_all_unlabel_data,
                        raw_pred_student,
                        instance_reg,
                    ) = self.model(
                        all_unlabel_data,
                        output_raw=True,
                        ignore_near=self.cfg.SEMISUPNET.PSEUDO_CLS_IGNORE_NEAR,
                        branch="unlabeled",
                    )
            else:
                record_all_unlabel_data, raw_pred_student, instance_reg = self.model(
                    all_unlabel_data,
                    output_raw=True,
                    ignore_near=self.cfg.SEMISUPNET.PSEUDO_CLS_IGNORE_NEAR,
                    branch="unlabeled",
                )

            new_record_all_unlabel_data = {}
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[
                    key
                ]
            record_dict.update(new_record_all_unlabel_data)

            # weight losses
            loss_loc_unsup_list = [
                "loss_fcos_loc_pseudo",
            ]
            loss_ctr_unsup_list = [
                "loss_fcos_ctr_pseudo",
            ]
            loss_cls_unsup_list = [
                "loss_fcos_cls_pseudo",
            ]
            loss_loc_sup_list = [
                "loss_fcos_loc",
            ]
            loss_ctr_sup_list = [
                "loss_fcos_ctr",
            ]
            loss_cls_sup_list = [
                "loss_fcos_cls",
            ]

            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    if (
                        key in loss_ctr_sup_list + loss_cls_sup_list
                    ):  # supervised classification + centerness loss
                        loss_dict[key] = record_dict[key] / (
                            self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT + 1.0
                        )
                    elif (
                        key in loss_ctr_unsup_list + loss_cls_unsup_list
                    ):  # unsupervised classifciation + centerness loss
                        loss_dict[key] = (
                            record_dict[key]
                            * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                            / (self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT + 1.0)
                        )

                    elif key in loss_loc_sup_list:  # supervised regression loss
                        loss_dict[key] = record_dict[key] / (
                            self.cfg.SEMISUPNET.UNSUP_REG_LOSS_WEIGHT + 1.0
                        )
                    elif key in loss_loc_unsup_list:  # unsupervised regression loss
                        loss_dict[key] = (
                            record_dict[key]
                            * self.cfg.SEMISUPNET.UNSUP_REG_LOSS_WEIGHT
                            / (self.cfg.SEMISUPNET.UNSUP_REG_LOSS_WEIGHT + 1.0)
                        )

                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] / (
                            self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT + 1.0
                        )

            if self.cfg.SOLVER.AMP.ENABLED:
                with autocast():
                    losses = sum(loss_dict.values())
            else:
                losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        if self.cfg.SOLVER.AMP.ENABLED:
            self._trainer.grad_scaler.scale(losses).backward()
            self._trainer.grad_scaler.step(self.optimizer)
            self._trainer.grad_scaler.update()
        else:
            losses.backward()
            self.optimizer.step()

    def _write_metrics(self, metrics_dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] * (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(self.cfg, self.model_teacher)
            return self._last_eval_results_teacher

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results_teacher))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator, cfg)
            # results_i = inference_on_dataset(model, data_loader, evaluator)

            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info(
                    "Evaluation results for {} in csv format:".format(dataset_name)
                )
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


# Unbiased Teacher Trainer for Faster RCNN
class UBRCNNTeacherTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        
        #Precisamos os datos sen augmentations para crear o banco de prototipos no PCB
        data_loader,label_noAugmentations,unlabel_noAugmentations = self.build_train_loader(cfg)

        # create an student model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        
        self.listaIds=[]
        self.listaImages=[]
        
        #Limiar flexible de flexmatch: creanse as listas necesarias vacias 
        
        if cfg.SEMISUPNET.FLEXIBLE_THRESHOLD:                
            #As listas son listas de listas (unha lista para cada id imaxe)
            #Para cada id imaxe, numero de instancias por clase que pasaron o limiar
            self.flexmatch_instances_per_class=[] #lista total
            self.iter_flexmatch_list=[] #lista da iteracion actual (necesaria para cada gpu)
            #Lista co id da imaxe que esta en cada posicion
            self.flexmatch_instances_per_class_idimages=[] #lista total
            self.iter_flexmatch_list_idimages=[] #lista da iteracion actual

            
        #Creamos o pcb: crease o banco de prototipos
        self.pcb = None
        if cfg.TEST.PCB_ENABLE:
            logger = logging.getLogger(__name__)
            logger.info("Start initializing PCB module, please wait...")
            #Recibe datos etiquetados sen augmentations
            data_loader_pcb=self.build_pcb_loader(cfg,label_noAugmentations)
            self.pcb = PrototypicalCalibrationBlock(cfg,data_loader=data_loader_pcb)

        # create an teacher model
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # Ensemble teacher and student model is for model saving and loading
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        
        #Se se reduce o numero de datos empregados, reducese o numero de iteracions
        if cfg.DATALOADER.REDUCE_UNLABEL_DATA:
            self.max_iter=int(self.max_iter*0.01*cfg.DATALOADER.UNLABEL_PERCENT_USE)
            
        
        #Estimacion factor de balanceo e numero de obxectos por clase para implementar limiar flexible
        if cfg.SEMISUPNET.FLEXIBLE_THRESHOLD: 
            num_labeled_per_class=np.zeros(80,dtype=np.int64)
            dataset_images=data_loader.label_dataset.dataset
            #Contaremos o numero de obxectos totais e dividimos entre numero de imaxes para ter una estimacion do numero de obxectos por imaxe
            totalImages=len(dataset_images)
            for image in dataset_images:
                clases=image[0]['instances'].gt_classes.numpy()
                for c in clases:
                    num_labeled_per_class[c]+=1
            #Numero de obxectos da clase que mais ten
            clase_max=np.max(num_labeled_per_class)

            ponderacions=num_labeled_per_class/clase_max
            ponderacions=np.divide(np.ones(80),ponderacions).astype(int)
            self.ponderacions_flexmatch=ponderacions
            
            #Obtemos o total de obxectos (tendo en conta as ponderacions)
            numObxPorClase_ponderado=[a * b for a, b in zip(num_labeled_per_class,ponderacions)]
            numObx_ponderado=np.sum(numObxPorClase_ponderado)
            numObxPorImaxe_ponderado=numObx_ponderado/totalImages
            #Estimacion numero de obxectos totais non etiquetados
            self.numObxUnlabel_estim=numObxPorImaxe_ponderado*len(data_loader.unlabel_dataset.dataset)
        
        #Gardanse os limiares o arquivo limiarflex.txt
        if comm.is_main_process():
            self.flex_file=open(cfg.OUTPUT_DIR+"/limiarflex.txt","a")
            self.flex_file.close()
             
        self.cfg = cfg
        
        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        if cfg.TEST.EVALUATOR == "COCOeval":
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        else:
            raise ValueError("Unknown test evaluator.")

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)
    
    @classmethod
    def build_pcb_loader(cls,cfg,label_dataset):
        mapper=DatasetMapperPCB(cfg,True)
        return build_detection_pcb_loader(cfg=cfg,label_dataset=label_dataset,mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()

                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step_full_semisup()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    # =====================================================
    # ================== Pseduo-labeling ==================
    # =====================================================
    #Modificase para implementar limiar flexible
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih",pseudo_label_method="thresholding"):
        #Empregase para indicar se a proposta se vai ignorar na rcnn
        proposal_bbox_inst.ignore=torch.zeros(size=proposal_bbox_inst.scores.shape,dtype=torch.bool,device=proposal_bbox_inst.scores.device)
        if proposal_type == "roih":
            if pseudo_label_method=="thresholding":
                valid_map = proposal_bbox_inst.scores > thres
                
                if self.cfg.SEMISUPNET.IGNORE_REGIONS:
                    valid_map = proposal_bbox_inst.scores > self.cfg.SEMISUPNET.THRESHOLD_IGNORE_REGIONS
                    for i in range(len(proposal_bbox_inst.scores)):
                        if proposal_bbox_inst.scores[i]<thres and proposal_bbox_inst.scores[i]>self.cfg.SEMISUPNET.THRESHOLD_IGNORE_REGIONS:
                            proposal_bbox_inst.ignore[i]=True
            elif pseudo_label_method=="flexible_thresholding":
                #Lo inicializo con todo falso
                valid_map=proposal_bbox_inst.scores > 1
                pred_classes=proposal_bbox_inst.pred_classes.cpu()
                thresholds=thres
                
                #Se non se configura un limiar maximo con selección aleatoria
                if not self.cfg.SEMISUPNET.RANDOM_MAX_THRESHOLD:

                    for i in range(len(pred_classes)):
                        clas=pred_classes[i]
                        if proposal_bbox_inst.scores[i]>thresholds[clas] and proposal_bbox_inst.scores[i]>self.cfg.SEMISUPNET.MIN_BBOX_THRESHOLD:
                            valid_map[i]=True
                        #Marcase como ignore se esta por debaixo do limiar flexible pero por enriba de self.cfg.SEMISUPNET.THRESHOLD_IGNORE_REGIONS
                        elif self.cfg.SEMISUPNET.IGNORE_REGIONS and proposal_bbox_inst.scores[i]>self.cfg.SEMISUPNET.THRESHOLD_IGNORE_REGIONS:
                            valid_map[i]=True
                            proposal_bbox_inst.ignore[i]=True
                else:
                    #Para cada clase, conto cantas instancias estan por enriba do limiar maximo e do seu limiar dinamico
                    proposals_maiorLimiarMax_maiorLimiarFlex=np.zeros(80,dtype=np.int32)
                    #Para cada clase unha lista con todas as propostas que pasen o maximo (guardo o seu indice)
                    indices_maiorLimiarMax={}
                    for i in range(len(pred_classes)):
                        clas=int(pred_classes[i])
                        if proposal_bbox_inst.scores[i]>thresholds[clas] and proposal_bbox_inst.scores[i]>self.cfg.SEMISUPNET.MIN_BBOX_THRESHOLD:
                            if proposal_bbox_inst.scores[i]>=self.cfg.SEMISUPNET.MAX_BBOX_THRESHOLD:
                                #Se esta por enriba do maximo e do limiar dinamico aumento a conta                 
                                proposals_maiorLimiarMax_maiorLimiarFlex[clas]+=1
                            else:
                                valid_map[i]=True
                        if proposal_bbox_inst.scores[i]>=self.cfg.SEMISUPNET.MAX_BBOX_THRESHOLD:
                            #Si esta por encima del maximo guardo el indice
                            if clas in indices_maiorLimiarMax:
                                indices_maiorLimiarMax[clas].append(i)
                            else:
                                indices_maiorLimiarMax[clas]=[i]
                    #Elixo aleatoriamente que propuestas con umbral maior que el maximo pasan para cada clase
                    for clase,listaIndices in indices_maiorLimiarMax.items():
                        numElegir=proposals_maiorLimiarMax_maiorLimiarFlex[clase]
                        random_indexes = random.sample(listaIndices, numElegir)
                        for i in random_indexes:
                            valid_map[i]=True



            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

            if proposal_bbox_inst.has("pred_boxes_std"):
                new_proposal_inst.pred_boxes_std = proposal_bbox_inst.pred_boxes_std[
                    valid_map, :
                ]
            if proposal_bbox_inst.has("ignore"):
                new_proposal_inst.ignore = proposal_bbox_inst.ignore[valid_map]
        else:
            raise ValueError("Error in proposal type.")

        return new_proposal_inst

    def process_pseudo_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            # thresholding
            if psedo_label_method == "thresholding" or psedo_label_method == "flexible_thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type,pseudo_label_method=psedo_label_method
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data
    
    #Actualiza self.iter_flexmatch_list e self.iter_flexmatch_list_idimages coa lista de instancias de cada clase que pasaron o limiar nesta iteracion (as imaxes estan en unlabel_data)
    def update_iter_flexmatch_list(self,unlabel_data):
        self.iter_flexmatch_list=[]
        self.iter_flexmatch_list_idimages=[]
        num_clases=self.cfg.MODEL.FCOS.NUM_CLASSES
        for im in unlabel_data:
            #Engadense as prediccions da imaxe
            im_clases=[]
            for i in range(len(im['instances'].gt_classes)):
                #Se non se ignora usase
                if not im['instances'].ignore[i]:
                    im_clases.append(int(im['instances'].gt_classes[i]))

            im_id=im['image_id']
            self.iter_flexmatch_list_idimages.append(im_id)
            self.iter_flexmatch_list.append(im_clases)


    
#Devuelve un array co novo limiar para cada clase.
#fixed_threshold: limiar fixo usado para calcular o limiar dinamico
    def obtain_flexible_threshold(self,fixed_threshold):
        num_clases=self.cfg.MODEL.FCOS.NUM_CLASSES
        clases=np.zeros(num_clases,dtype=np.int64).tolist()
        
        #Convierto a miña lista de listas nunha unica lista
        listaClases=[item for sublist in self.flexmatch_instances_per_class for item in sublist]
        #Contase cantas instancias de cada clase hai gardadas en flexmatch_instances_per_class
        for c in listaClases:
            clases[c]+=1

        #Empreganse as ponderacions calculadas ao inicio
        ponderacions=self.ponderacions_flexmatch
        clases=[a * b for a, b in zip(clases,ponderacions )]    
        clase_max=np.max(clases)

        #Array que garda o novo limiar de cada clase
        thresholds=[]
        if not self.cfg.SEMISUPNET.FLEX_WARMUP:
            if clase_max==0:
                return np.full(num_clases,0)
            for i in range(num_clases):           
                #Beta de flexmatch sen warmup
                beta=clases[i]/clase_max
                thresholds.append(beta*fixed_threshold)
        else:       
            numobxectosUsados=np.sum(clases)
            numObxSinUsar=self.numObxUnlabel_estim-numobxectosUsados
            #Se dominan os datos sen utilizar, etapa de warmup
            if clase_max<numObxSinUsar:
                
                for i in range(num_clases):
                    #Beta etapa warm up
                    beta=clases[i]/numObxSinUsar
                    th=beta*fixed_threshold
                    thresholds.append(th)
            else:            
                for i in range(num_clases):
                    #Beta de flexmatch sen warmup
                    beta=clases[i]/clase_max
                    thresholds.append(beta*fixed_threshold)

        return thresholds
        

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    def run_step_full_semisup(self):
            
        self._trainer.iter = self.iter
        assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
        data_time = time.perf_counter() - start

        # burn-in stage (supervised training with labeled data)
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:

            semisup=False

            # input both strong and weak supervised data into model
            if self.cfg.SEMISUPNET.USE_SUP_STRONG == "both":
                all_label_data = label_data_q + label_data_k
            else:
                all_label_data = label_data_k

            record_dict, _, _, _ = self.model(all_label_data, branch="supervised")

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = record_dict[key]
            losses = sum(loss_dict.values())

        else:
            semisup=True
            # copy student model to teacher model
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                self._update_teacher_model(keep_rate=0.0)

            if (
                self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
            ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
                cur_ema_rate = self.cfg.SEMISUPNET.EMA_KEEP_RATE
                self._update_teacher_model(keep_rate=cur_ema_rate)
                
            record_dict = {}
            record_dict["EMA_rate"] = cur_ema_rate

            #  generate the pseudo-label using teacher model
            # note that we do not convert to eval mode, as 1) there is no gradient computed in
            # teacher model and 2) batch norm layers are not updated as well
            with torch.no_grad():
                (
                    _,
                    proposals_rpn_unsup_k,
                    proposals_roih_unsup_k,
                    _,
                ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")
            
            
            #Modificanse as confianzas na clasificacion do teacher usando o PCB    
            if self.cfg.TEST.PCB_ENABLE:
                for i in range(0,len(proposals_roih_unsup_k)):
                    proposals_roih_unsup_k[i] = self.pcb.execute_calibration(unlabel_data_k[i], proposals_roih_unsup_k[i])
            
            #Se esta nunha etapa previa a aplicar limiar dinamico as instancias obteñense igual para estimar os limiares mellor despois
            if self.cfg.SEMISUPNET.FLEXIBLE_THRESHOLD and self.iter<=self.cfg.SEMISUPNET.NUM_ITERS_BEFORE_FLEX:
                comm.synchronize()
                #Lista conjunta de esta iteracion
                lista=comm.all_gather(self.iter_flexmatch_list)
                lista_idimages=comm.all_gather(self.iter_flexmatch_list_idimages)
                for i in range(len(lista_idimages)):
                        l=lista_idimages[i]
                        for j in range(len(l)):
                            id=l[j]
                            if id in self.flexmatch_instances_per_class_idimages:
                                index=self.flexmatch_instances_per_class_idimages.index(id)
                                self.flexmatch_instances_per_class[index]=lista[i][j]
                                
                            else:
                                self.flexmatch_instances_per_class_idimages+=[id]
                                self.flexmatch_instances_per_class.append(lista[i][j])
            
            if self.cfg.SEMISUPNET.FLEXIBLE_THRESHOLD and self.iter>self.cfg.SEMISUPNET.NUM_ITERS_BEFORE_FLEX:
                
                #Sincronizanse as gpus para obter a lista total de instancias de cada clase 
                comm.synchronize()
                #Listas conxuntas de esta iteracion
                lista=comm.all_gather(self.iter_flexmatch_list)
                lista_idimages=comm.all_gather(self.iter_flexmatch_list_idimages)
                #Cada 500 iteracions almacenanse os limiares para cada clase
                if self.iter%500==0:                 
                    if comm.is_main_process(): 
                        self.flex_file=open(self.cfg.OUTPUT_DIR+"/limiarflex.txt","a")
                        umbral=self.obtain_flexible_threshold(self.cfg.SEMISUPNET.BBOX_THRESHOLD)
                        self.flex_file.write("Limiares\n"+" ".join(str(x) for x in umbral)+"\n")
                        self.flex_file.close()
                          
                #Actualizanse as listas de instancias de cada clase que pasaron o limiar 
                #coa informacion da ultima iteracion
                for i in range(len(lista_idimages)):
                    l=lista_idimages[i]
                    for j in range(len(l)):
                        id=l[j]
                        if id in self.flexmatch_instances_per_class_idimages:
                            index=self.flexmatch_instances_per_class_idimages.index(id)
                            self.flexmatch_instances_per_class[index]=lista[i][j]
                            
                        else:
                            self.flexmatch_instances_per_class_idimages+=[id]
                            self.flexmatch_instances_per_class.append(lista[i][j])
                        
                #Obtense o vector de limiares para cada clase
                cur_threshold=self.obtain_flexible_threshold(self.cfg.SEMISUPNET.BBOX_THRESHOLD)

                # Procesanse as pseudoetiquetas co vector de limiares
                pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
                    proposals_roih_unsup_k, cur_threshold, "roih", "flexible_thresholding"
                )
            else:
                #Se non hai limiar flexible o limiar e un numero
                cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD
                # Pseudo_labeling for ROI head (bbox location/objectness)
                pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
                    proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
                )
            
            
            joint_proposal_dict = {}
            joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k

            #  add pseudo-label to unlabeled data
            unlabel_data_q = self.remove_label(unlabel_data_q)
            unlabel_data_k = self.remove_label(unlabel_data_k)

            unlabel_data_q = self.add_label(
                unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
            )
            unlabel_data_k = self.add_label(
                unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
            )

            if self.cfg.SEMISUPNET.USE_SUP_STRONG == "both":
                all_label_data = label_data_q + label_data_k
            else:
                all_label_data = label_data_k

            all_unlabel_data = unlabel_data_q

            
            if self.cfg.SEMISUPNET.FLEXIBLE_THRESHOLD and self.cfg.SEMISUPNET.FLEXMATCH_NMS and self.iter>self.cfg.SEMISUPNET.NUM_ITERS_BEFORE_FLEX:
            #Aplicamos NMS antes de pasarlle o gt ao Student
                for dat in all_unlabel_data:
                    boxes=dat["instances"].gt_boxes.tensor
                    scores=dat["instances"].scores
                    nms_result=torchvision.ops.nms(boxes,scores,self.cfg.SEMISUPNET.IOU_FLEXMATCH_NMS)
                    num_boxes=len(dat["instances"].gt_boxes.tensor)
                    map=torch.zeros(num_boxes,dtype=torch.bool)
                    for i in range(num_boxes):
                        if i in nms_result:
                            map[i]=True
                            
                    dat["instances"].gt_boxes.tensor=dat["instances"].gt_boxes.tensor[map]
                    dat["instances"].scores=dat["instances"].scores[map]
                    dat["instances"].gt_classes=dat["instances"].gt_classes[map]                    
                    dat["instances"].pred_boxes_std=dat["instances"].pred_boxes_std[map]
            
            #Actualizamos as listas coas novas instancias que pasaron o limiar
            if self.cfg.SEMISUPNET.FLEXIBLE_THRESHOLD:
                self.update_iter_flexmatch_list(all_unlabel_data)
                
            record_all_label_data, _, _, _ = self.model(
                all_label_data, branch="supervised"
            )
            record_dict.update(record_all_label_data)

            record_all_unlabel_data, _, _, _ = self.model(
                all_unlabel_data, branch="unsup_data_train"
            )

            new_record_all_unlabel_data = {}
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[
                    key
                ]
            record_dict.update(new_record_all_unlabel_data)

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    if key == "loss_rpn_loc_pseudo":
                        # pseudo RPN bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 0
                    elif key == "loss_box_reg_pseudo":
                        # pseudo ROIhead box regression
                        loss_dict[key] = (
                            record_dict[key] * self.cfg.SEMISUPNET.UNSUP_REG_LOSS_WEIGHT
                        )
                    elif key[-6:] == "pseudo":
                        # pseudo RPN, ROIhead classification
                        loss_dict[key] = (
                            record_dict[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                        )
                    else:  # supervised loss
                        loss_dict[key] = record_dict[key]

            losses = sum(loss_dict.values())


        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()         
        

    def _write_metrics(self, metrics_dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():

                new_teacher_dict[key] = (
                    student_model_dict[key] * (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(self.cfg, self.model_teacher)
            return self._last_eval_results_teacher

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results_teacher))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret
