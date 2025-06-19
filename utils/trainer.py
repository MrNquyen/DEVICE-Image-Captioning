import torch
import warnings
import os
import numpy as np

from torch import nn
import torch.nn.functional as F
from utils.configs import Config
from projects.models.device_model import DEVICE
from datasets.dataset import get_loader
from utils.model_utils import get_optimizer_parameters
from utils.module_utils import _batch_padding, _batch_padding_string
from utils.logger import Logger
from utils.vocab import OCRVocab

# ~Trainer~
class Trainer():
    def __init__(self, config, args, run_type):
        self.args = args
        self.config = Config(config)
        self.run_type = run_type
        self.model = DEVICE(self.config.config_model)
        self.writer = Logger()
        self.writer_evaluation = Logger()
        self.writer_inference = Logger()

    #---- LOAD TASK
    def load_task(self):
        batch_size = self.config.config_training["batch_size"]
        self.train_loader = get_loader(dataset_config=self.config.config_dataset, batch_size=batch_size, split="train")
        self.val_loader = get_loader(dataset_config=self.config.config_dataset, batch_size=batch_size, split="val")
        self.test_loader = get_loader(dataset_config=self.config.config_dataset, batch_size=batch_size, split="test")

    #---- BUILD
    def build_training_params(self):
        self.max_epochs = self.config.config_training["epochs"]
        self.batch_size = self.config.config_training["batch_size"]
        self.max_iterations = self.config.config_training["max_iterations"]
        self.current_iteration = 0
        self.current_epoch = 0

        # Training
        self.optimizer = self.build_optimizer()
        self.loss_fn = self.build_loss()
        self.lr_scheduler = self.build_scheduler()

        # Resume training
        if self.args.resume_file != None:
            self.load_model(self.args.resume_file)


    def build_scheduler(self, optimizer, config_lr_scheduler):
        if not config_lr_scheduler["status"]:
            return None
        lr_scheduler = nn.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=config_lr_scheduler["step_size"],
            gamma=config_lr_scheduler["gamma"]
        )
        return lr_scheduler


    def build_loss(self):
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn


    def build_optimizer(self, model, config_optimizer):
        if not hasattr(config_optimizer, "type"):
            raise ValueError(
                "Optimizer attributes must have a 'type' key "
                "specifying the type of optimizer. "
                "(Custom or PyTorch)"
            )
        optimizer_type = config_optimizer.type

        #-- Load params
        if not hasattr(config_optimizer, "params"):
            warnings.warn(
                "optimizer attributes has no params defined, defaulting to {}."
            )
        optimizer_params = getattr(config_optimizer, "params", {})
        
        #-- Load optimizer class
        if not hasattr(torch.optim, optimizer_type):
            raise ValueError(
                "No optimizer found in torch.optim"
            )
        optimizer_class = getattr(torch.optim, optimizer_type)
        parameters = get_optimizer_parameters(
            model=model, 
            config=config_optimizer
        )
        optimizer = optimizer_class(parameters, **optimizer_params)
        return optimizer
    

    #---- STEP
    def _forward_pass(self, batch):
        """
            Forward to model
        """
        scores_output = self.model(batch)
        return scores_output


    def _extract_loss(self, scores, targets):
        loss_output = self.loss_fn(
            scores, targets
        ) 
        return loss_output

    def _backward(self, loss):
        """
            Backpropagation
        """
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        self._run_scheduler()
    

    def _run_scheduler(self):
        """
            Learning rate scheduler
        """
        # self.lr_scheduler.step(self.current_iteration)
        self.lr_scheduler.step()


    def _backward(self, loss):
        """
            Backpropagation
        """
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        self._run_scheduler()


    #---- MODE
    def preprocess_batch(self, batch):
        """
            Function:
                - Padding ocr and obj to the same length
                - Create mask for ocr and obj
        """
        model_config = self.model.config.config_model
        box_pad = torch.rand((1, 4))
        # Padding ocr
        ocr_feat_pad = torch.rand((1, self.dim_ocr))
        batch["list_ocr_boxes"], ocr_mask = _batch_padding(batch["list_ocr_boxes"], max_length=model_config["ocr"]["num_ocr"], pad_value=box_pad)
        batch["list_ocr_feat"] = _batch_padding(batch["list_ocr_feat"], max_length=model_config["ocr"]["num_ocr"], pad_value=ocr_feat_pad, return_mask=False)
        batch["list_ocr_tokens"] = _batch_padding_string(batch["list_ocr_tokens"], max_length=model_config["ocr"]["num_ocr"], pad_value="<pad>", return_mask=False)
        batch["ocr_mask"] = ocr_mask

        # Padding obj
        obj_feat_pad = torch.rand((1, self.dim_obj))
        batch["list_obj_boxes"], obj_mask = _batch_padding(batch["list_obj_boxes"], max_length=model_config["obj"]["num_obj"], pad_value=box_pad)
        batch["list_obj_feat"] = _batch_padding(batch["list_obj_feat"], max_length=model_config["obj"]["num_obj"], pad_value=obj_feat_pad, return_mask=False)
        batch["obj_mask"] = obj_mask
        return batch


    #---- MODE
    def train(self):
        self.writer.LOG_INFO("=== Model ===")
        self.writer.LOG_INFO(self.model)

        self.writer.LOG_INFO("Starting training...")
        self.model.train()

        while self.current_iteration < self.max_iterations:
            self.current_epochs += 1
            for batch in self.train_loader:
                batch = self.preprocess_batch()
                self.current_iteration += 1
                scores_output = self._forward_pass(batch)
                loss = self._extract_loss(scores_output)
                self._backward(loss)
                
                if self.current_iteration < self.max_iterations:
                    break
            
            if self.current_epoch % 2 == 0:
                self.evaluate(epoch_id=self.current_epoch)

    
    def evaluate(self, epoch_id=None):
        with torch.inference_mode():
            self.model.eval()
            for batch in self.val_loader:
                batch = self.preprocess_batch()
                scores_output = self._forward_pass(batch)
                if not epoch_id==None:
                    self.writer_evaluation.LOG_INFO(f"Logging at epoch {epoch_id}")
            
            # Turn on train mode to continue training
            self.model.train()
        return scores_output
    

    def test(self):
        with torch.inference_mode():
            self.model.eval()
            for batch in self.test_loader:
                batch = self.preprocess_batch()
                scores_output = self._forward_pass(batch)
                self.writer_evaluation.LOG_INFO(f"Logging at epoch {epoch_id}")

        return scores_output


    def inference(self, mode):
        """
            Parameters:
                mode:   Model to run "val" or "test"
        """
        if mode=="val":
            self.writer_inference.LOG_INFO("=== Inference Validation Split ===")
            scores_output = self.evaluate()
        elif mode=="test":
            self.writer_inference.LOG_INFO("=== Inference Test Split ===")
            scores_output = self.test()
        else:
            self.writer_inference.LOG_ERROR(f"No mode available for {mode}")
            return

    #---- FINISH
    def save_model(self, model, loss, optimizer, epoch):
        if os.path.exists(self.args.save_dir):
            self.writer("Save dir not exist")
            raise FileNotFoundError
        
        model_path = os.path.join(self.args.save_dir, f"model_{epoch}.pth")
        self.writer.LOG_DEBUG(f"Model save at {model_path}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, model_path)

    
    def load_model(self, model_path):
        checkpoint = torch.load(model_path, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        self.writer.LOG_INFO(f"=== Load model at epoch: {epoch} || loss: {loss} ===")


    def get_pred_captions(self, scores, ocr_tokens):
        """
            Predict batch
        """
        # Get logits
        pred_logits = F.softmax(scores, dim=-1)
        pred_ids = np.argmax(pred_logits, axis=1)

        # Captioning
        common_vocab = self.model.word_embedding.common_vocab
        vocab_size = common_vocab.get_size()
        ocr_vocab_object = OCRVocab(ocr_tokens=ocr_tokens)
        captions_pred = [
            " ".join([
                common_vocab.get_idx_word(idx)
                if idx < vocab_size
                else ocr_vocab_object[i].get_idx_word(idx - vocab_size)
                for idx in item_pred_ids
            ])
            for i, item_pred_ids in enumerate(pred_ids)
        ]
        return captions_pred # BS, 
