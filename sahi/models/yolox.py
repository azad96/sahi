
import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.import_utils import check_requirements


import logging
import warnings
from typing import Dict, List, Optional, Union
import importlib
import torch
import numpy as np
from yolox.data.datasets import COCO_CLASSES

from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.cv import get_bbox_from_bool_mask

from yolox.utils import postprocess
from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp

class YoloXDetectionModel(DetectionModel):
    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """
        try:
            import yolox
        except ImportError:
            raise ImportError("Please run pip install -U yolox")
        # current_exp = importlib.import_module(self.config_path)
        # exp = current_exp.Exp()
        exp = get_exp(self.config_path, None)
        
        model = exp.get_model()
        # model.cuda()
        model.to(self.device)
        model.eval()
        #print(model)
        self.model = model
        ckpt = torch.load(self.model_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])

        if not self.category_mapping:
              category_mapping = {str(ind): category_name for ind, category_name in enumerate(self.category_names)}
              self.category_mapping = category_mapping

    def perform_inference(self, image: np.ndarray, image_size: int = None):
        try:
            import yolox
        except ImportError:
            raise ImportError('Please run "pip install -U yolox" ' "to install YOLOX first for YOLOX inference.")

        # Confirm model is loaded
        assert self.model is not None, "Model is not loaded, load it by calling .load_model()"

        preproc = ValTransform(legacy = False)
        if image_size is not None:
            tensor_img, _ = preproc(image, None, image_size)
        elif self.image_size is not None:
            tensor_img, _ = preproc(image, None, self.image_size)
        else:
            tensor_img, _ = preproc(image, None, (256,256))
        
        tensor_img = torch.from_numpy(tensor_img).unsqueeze(0)
        tensor_img = tensor_img.float()
        # tensor_img = tensor_img.cuda()
        tensor_img.to(self.device)

        with torch.no_grad():
            prediction_result = self.model(tensor_img)
            prediction_result = postprocess(
                    prediction_result, len(self.category_names), self.confidence_dict,
                    self.nms_threshold
                )
        
        if (prediction_result[0] is not None):
            prediction_result = prediction_result[0].cpu()
            bboxes = prediction_result[:,0:4]
            if image_size is not None:
                bboxes /= min(image_size[0] / image.shape[0], image_size[1] / image.shape[1])
            elif self.image_size is not None:
                bboxes /= min(self.image_size[0] / image.shape[0], self.image_size[1] / image.shape[1])
            else:
                bboxes /= min(256 / image.shape[0], 256 / image.shape[1])

            prediction_result[:,0:4] = bboxes

        self._original_predictions = prediction_result

    @property
    def category_names(self):
        return self.classes

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions
        bboxes=[]
        bbclasses=[]
        scores=[]

        if isinstance(shift_amount_list[0], int):
            shift_amount_list = [shift_amount_list]
        if full_shape_list is not None and isinstance(full_shape_list[0], int):
            full_shape_list = [full_shape_list]
        
        if(original_predictions[0] is not None):
            bboxes = original_predictions[:,0:4]
            bbclasses = original_predictions[:, 6]
            scores = original_predictions[:, 4] * original_predictions[:, 5]        

        shift_amount = shift_amount_list[0]
        full_shape = None if full_shape_list is None else full_shape_list[0]

        object_prediction_list_per_image = []
        object_prediction_list = []

        for ind in range(len(bboxes)):
              box = bboxes[ind]
              cls_id = int(bbclasses[ind])
              score = scores[ind]
              if score < self.confidence_dict[cls_id]:
                continue
              
              x0 = int(box[0])
              y0 = int(box[1])
              x1 = int(box[2])
              y1 = int(box[3])

              bbox = [x0,y0,x1,y1]

              object_prediction = ObjectPrediction(
                bbox = bbox,
                category_id=cls_id,
                bool_mask=None,
                category_name=self.category_mapping[str(cls_id)],
                shift_amount=shift_amount,
                score=score,
                full_shape=full_shape,
            )
              object_prediction_list.append(object_prediction)
        
        object_prediction_list_per_image = [object_prediction_list]
        self._object_prediction_list_per_image = object_prediction_list_per_image