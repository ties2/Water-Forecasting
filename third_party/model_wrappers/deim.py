import os
import sys
import types

import torch

# mock faster_coco_eval so we don't need it as dependency just for DEIM
try:
    import faster_coco_eval
except:
    fce_module = types.ModuleType('faster_coco_eval')
    sys.modules['faster_coco_eval'] = fce_module
    fce_module.core = types.ModuleType('faster_coco_eval.core')
    sys.modules['faster_coco_eval.core'] = fce_module.core
    fce_module.core.mask = types.ModuleType('faster_coco_eval.core.mask')
    sys.modules['faster_coco_eval.core.mask'] = fce_module.core.mask

    fce_module.init_as_pycocotools = lambda : None
    fce_module.COCO = None
    fce_module.COCOeval_faster = None

# mock calflops so we don't need it as dependency just for DEIM
try:
    import calflops
except:
    calflops_module = types.ModuleType('calflops')
    sys.modules['calflops'] = calflops_module

    calflops_module.calculate_flops = lambda : None


try:
    import third_party.DEIM as DEIM
    import third_party.DEIM.engine.core as DEIMCore
except ModuleNotFoundError:
    print("!"*120)
    print("Failed to import third_party.DEIM, git submodule missing?")
    print("Try running 'git submodule update --init --recursive' in this project's root folder.")
    print("!"*120)
    raise

from common.data.datatypes import BoundingBox

# DEIM.__file__

_deim_path = os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0], "DEIM", "configs", "deim_dfine")
DEIM_CONFIGS = {
    "deim_n": os.path.join(_deim_path, "deim_hgnetv2_n_coco.yml"),
    "deim_s": os.path.join(_deim_path, "deim_hgnetv2_s_coco.yml"),
    "deim_m": os.path.join(_deim_path, "deim_hgnetv2_m_coco.yml"),
    "deim_l": os.path.join(_deim_path, "deim_hgnetv2_l_coco.yml"),
    "deim_x": os.path.join(_deim_path, "deim_hgnetv2_x_coco.yml"),
}

DEIM_HGNET_NAMES = {
    "deim_n": "B0",
    "deim_s": "B0",
    "deim_m": "B2",
    "deim_l": "B4",
    "deim_x": "B5",
}

DEIM_HGNET_URLS = {
    'B0': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth',
    'B1': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B1_stage1.pth',
    'B2': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B2_stage1.pth',
    'B3': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B3_stage1.pth',
    'B4': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth',
    'B5': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B5_stage1.pth',
    'B6': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B6_stage1.pth',
}



def simple_nms_perclass(boxes: list[BoundingBox], maxdist: float) -> list[BoundingBox]:
    """
    Really basic non-max suppression, applied per class
    Calls 'simple_nms' for each class with only the boxes belonging to that class.
    """
    # find classes
    boxes_per_class: dict[int, list[BoundingBox]] = {}
    for box in boxes:
        if box._class_id not in boxes_per_class:
            boxes_per_class[box._class_id] = []
        boxes_per_class[box._class_id].append(box)

    # call simple_nms for each class
    result_boxes: list[BoundingBox] = []
    for boxes in boxes_per_class.values():
        ok_boxes = simple_nms(boxes, maxdist=maxdist)
        result_boxes.extend(ok_boxes)

    # return result
    return result_boxes

def simple_nms(boxes: list[BoundingBox], maxdist: float) -> list[BoundingBox]:
    """
    Really basic non-max suppression
    If two boxes are within [maxdist] pixels of each other (as measured from the centre of the boxes), the box with the lower confidence will be removed.
    Should the confidences be exactly the same, the box which comes later in the list will be removed.
    """
    ok_boxes = []
    for box_i, box in enumerate(boxes):
        boxok = True

        for box2_i, box2 in enumerate(boxes):
            if box2 is box:     # skip self
                continue

            dx = box2.x - box.x
            dy = box2.y - box.y
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if dist <= maxdist and (box2._confidence > box._confidence or (box2._confidence == box._confidence and box_i > box2_i)):
                boxok = False
                break

        if boxok:
            ok_boxes.append(box)
    return ok_boxes

class DeimDetector:
    _local_model_dir = os.path.join("DEIM_pretrained_weights", "hgnetv2")

    def __init__(self, device: torch.device):
        super().__init__()

        self._device = device

        self._model_type: str = ""
        self._resolution: tuple[int, int] = (-1, -1)
        self._num_classes: int = -1

        self.model: torch.nn.Module | None = None

        self.optimiser: DEIM.optim.Optimizer | None = None
        self.criterion: torch.nn.Module | None = None
        self.postprocessor = None

    def _download_weights(self):
        """
        From DEIM source code, but with torch.distributed stripped. Work-around for DEIM pretrained weights downloads failing due to requiring torch.distributed being set up correctly, even though
        the rest of the code works fine without it.
        """
        RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
        HGNET_model = DEIM_HGNET_NAMES[self._model_type]
        HGNET_url = DEIM_HGNET_URLS[HGNET_model]
        try:
            model_path = os.path.join(self._local_model_dir,  'PPHGNetV2_' + HGNET_model + '_stage1.pth')
            if not os.path.exists(model_path):
                # If the file doesn't exist locally, download from the URL
                print(GREEN + "If the pretrained HGNetV2 can't be downloaded automatically. Please check your network connection." + RESET)
                print(GREEN + "Or download the model manually from " + RESET + f"{HGNET_url}" + GREEN + " to " + RESET + f"{self._local_model_dir}." + RESET)
                state = torch.hub.load_state_dict_from_url(HGNET_url, map_location='cpu', model_dir=self._local_model_dir)
                print(f"Downloaded stage1 {HGNET_model} HGNetV2 from URL.")

        except (Exception, KeyboardInterrupt) as e:
            raise RuntimeError(f"Failed to load pretrained HGNetV2 model. Download the model manually from {HGNET_url} to {self._local_model_dir}.")

    def create_model(self, model_type: str, num_classes: int, resolution: None | int | tuple[int, int] = None,  initial_lr: float = None, pretrained: bool = True, update_dict: dict = None) -> None:
        # preprocess parameters
        if isinstance(resolution, int):
            resolution = (resolution, resolution)

        # store params
        self._model_type = model_type
        self._resolution = resolution
        self._num_classes = num_classes

        # download pretrained weights
        if pretrained:
            self._download_weights()

        # create the DEIM model
        deim_cfg_path = DEIM_CONFIGS.get(model_type, None)
        if deim_cfg_path is None:
            raise RuntimeError(f"Invalid model specified: '{model_type}'. Must be one of {DEIM_CONFIGS.keys()}")

        # load DEIM model config and create model
        _update_dict = {'eval_spatial_size': list(resolution) if resolution is not None else None, 'num_classes': num_classes, 'val_dataloader': None, 'train_dataloader': None, 'remap_mscoco_category': False}
        if initial_lr is not None:
            _update_dict.update({
                'optimizer': {
                    'lr': initial_lr,
                    'weight_decay': initial_lr / 4,
                    'params': [{'params': '^(?=.*backbone)(?!.*bn).*$', 'lr': initial_lr / 2}, {'params': '^(?=.*(?:norm|bn)).*$', 'weight_decay': 0.}]
                }
            })
        _update_dict.update({
            "HGNetv2": {
                'pretrained': pretrained,
                'local_model_dir': self._local_model_dir + os.path.sep
            },
        })

        if update_dict is not None:
            _update_dict.update(update_dict)
        cfg = DEIMCore.YAMLConfig(deim_cfg_path, **_update_dict)

        # get model from config
        self.model = cfg.model.to(self._device)

        # store optimiser/criterion/scheduler
        self.optimiser = cfg.optimizer
        self.criterion = cfg.criterion.to(self._device)
        self.postprocessor = cfg.postprocessor
        self.postprocessor.deploy()

    def load_checkpoint(self, model_path: str, model_type: str = None, flexible_resolution: bool = False) -> dict:
        # load the checkpoint from disk
        trained_model = torch.load(model_path, map_location=self._device, weights_only=True)
        model_type = trained_model.get("deim_type", model_type)
        resolution = trained_model["resolution"] if not flexible_resolution else None

        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        num_classes = trained_model.get("num_classes", 3)

        # create model
        self.create_model(model_type, num_classes, pretrained=False, resolution=resolution)

        # load trained weights into model
        self.model.load_state_dict(trained_model['model'])

        # return metadata (or empty dict, if undefined)
        return trained_model.get("metadata", {})

    def save_checkpoint(self, model_path: str, metadata: dict = None):
        save_dict = {'model': self.model.state_dict(), 'deim_type': self._model_type, 'resolution': self._resolution, 'num_classes': self._num_classes}
        if metadata is not None:
            save_dict['metadata'] = metadata
        torch.save(save_dict, model_path)

    def decode_predictions(self, predictions, image_width, image_height, conf_th: float = 0.0, nms_maxdist: float = 0.01) -> list[list[BoundingBox]]:
        labels, boxes, scores = self.postprocessor.forward(predictions, torch.tensor([[image_width, image_height]], device=self._device))
        labels = labels.cpu()
        boxes = boxes.cpu()
        scores = scores.cpu()

        result_boxes = []
        for sample_labels, sample_boxes, sample_scores in zip(labels, boxes, scores):
            result_sample_boxes = []
            for label, box, score in zip(sample_labels, sample_boxes, sample_scores):
                if score < conf_th:
                    continue
                bbox = BoundingBox(int(label))
                bbox.set_minmax_xy(*box)
                bbox.set_confidence(float(score))
                result_sample_boxes.append(bbox)

            if nms_maxdist > 0.0:
                result_sample_boxes = simple_nms_perclass(result_sample_boxes, nms_maxdist * (image_height + image_width) / 2)

            result_boxes.append(result_sample_boxes)
        return result_boxes

    def forward(self, image: torch.Tensor, targets: list[dict] | None = None) -> dict:
        predictions = self.model.forward(image, targets=targets)

        # workaround for bug in DEIM criterion where it needs these keys to be defined even though they aren't actually used (for validation)
        if 'aux_outputs' not in predictions:
            predictions.update({'aux_outputs': [], 'enc_aux_outputs': [], 'enc_meta': {'class_agnostic': False}})

        return predictions

    def train(self):
        self.model.train()
        self.criterion.train()

    def eval(self):
        self.model.eval()
        self.criterion.eval()

    def deploy(self):
        self.model.deploy()

    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse)
