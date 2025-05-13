from import_library import *
from utils import *

class ODLoss(nn.Module):
    def __init__(self, C, lambda_conf, lambda_class, lambda_coord, lambda_noobj):
        super(ODLoss, self).__init__()

        self.C = C
        self.lambda_conf = lambda_conf
        self.lambda_class = lambda_class
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    
    @torch.cuda.amp.autocast()
    def forward(self, predictions, targets):
        batch_size = predictions.size(0)
        all_predictions = predictions.reshape(-1, predictions.size(-1)) # (batch, grid_num, 4) -> (batch*grid_num, 4)
        all_targets = targets.reshape(-1, targets.size(-1))

        # Index
        conf_idx = slice(self.C, self.C+1)
        box_idx = slice(self.C+1, self.C+3)
        class_idx = slice(0, self.C)

        # Confidence and Coordinate
        pred_conf = all_predictions[..., conf_idx]
        target_conf = all_targets[..., conf_idx]
        pred_box = all_predictions[..., box_idx]
        target_box = all_targets[..., box_idx]
        pred_class = all_predictions[..., class_idx]
        target_class = all_targets[..., class_idx]

        # Object mask
        obj_mask = target_conf > 0.5
        noobj_mask = ~obj_mask

        # Calculate Intersection over Union (IoU)
        iou = cal_iou(pred_box, target_box) 

        ### Coordinate loss ###
        coord_loss = ((1-iou) * obj_mask.float()).sum() / (obj_mask.sum() + 1e-8)

        ### Confidence loss ###
        obj_conf_loss = F.binary_cross_entropy_with_logits(pred_conf[obj_mask], target_conf[obj_mask], reduction='sum') / (obj_mask.sum() + 1e-8)
        noobj_conf_loss = F.binary_cross_entropy_with_logits(pred_conf[noobj_mask], target_conf[noobj_mask], reduction='sum') / (noobj_mask.sum() + 1e-8)
        conf_loss = (self.lambda_noobj * noobj_conf_loss) + obj_conf_loss

        ### Classification loss ###
        classification_loss = F.binary_cross_entropy_with_logits(pred_class[obj_mask], target_class[obj_mask], reduction='sum') / (obj_mask.sum() + 1e-8)
        
        ### Final_loss
        total_loss = (conf_loss * self.lambda_conf + coord_loss * self.lambda_coord + classification_loss * self.lambda_class)

        return total_loss, coord_loss, classification_loss, conf_loss