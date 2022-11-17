import cv2
import torch
from torch.utils import data
import torchvision
from torchvision import transforms

import numpy as np

import predict
import libs.transforms as transforms_parsing
import libs.polygon2mask as polygon2mask

torch.multiprocessing.set_start_method("spawn", force=True)
from loguru import logger
import torch.backends.cudnn as cudnn


class Model():

    def __init__(self):
        self.criterion = None
        self._ignore_value = None
        self.connect_predict = predict
        self.param = self.connect_predict.get_arguments()
        self.gpus = self.connect_predict.set_paralle_gpus(self.param)
        self.aspect_ratio = 735 * 1.0 / 490
        self.crop_size = np.asarray((735,490))

        #self.connect_predict.do_data_preparation(self.param)
        self.cnt = 0

    def modelInit(self):
        self.model, self.criterion, self.no_classes, self._ignore_value = self.connect_predict.init_model(self.param)
        return self.model

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    # --------------------------------------------------------------------------------------------------------------- #
    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)

        return center, scale

    def run(self):
        model = self.model
        toTensor = torchvision.transforms.ToTensor()
        device = torch.device('cpu')
        model.eval()
        #데이터셋 설정
        interp = torch.nn.Upsample(size=(735, 490), mode='bilinear', align_corners=True)
        # 무조건 하나만 받는다 전제 .
        parsing_preds = np.zeros((1, 735, 490),
                                 dtype=np.uint8)
        scales = np.zeros((1,2),dtype=np.float32)
        centers = np.zeros((1,2),dtype=np.int32)
        # PLAN B !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
        # data = self.connect_predict.realtime_predict_main(self.param)
        # logger.info(f'data {data} 도착 !!')
        idx = 0
        # 모델 로드 처리 되면 시작
        # 여기만 바뀌면 됨.
        raw = cv2.imread(filepath, cv2.IMREAD_COLOR)
        org_h,org_w,_ = raw.shape

        image = raw.copy()
        image = cv2.resize(image, (735, 490))
        num_images = 1
        batch_size = 1
        #-------------- 데이터셋.py
        h, w ,_ = image.shape
        center, scale = self._box2cs([0,0,org_w-1,org_h-1])
        r = 0

        trans = transforms_parsing.get_affine_transform(center, scale, r, self.crop_size)
        input_im = cv2.warpAffine(
            image,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        scales[idx:idx+num_images,:] = scale
        centers[idx:idx+num_images,:] = center

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,

        ])
        input_im = transform(input_im)
        input_im = input_im.unsqueeze(0)
        pred_dataset = input_im,_,_,_

        pred_loader = data.DataLoader(pred_dataset,
                                      batch_size=batch_size * len(self.gpus),
                                      shuffle=True, num_workers=2,
                                      pin_memory=True)

        with torch.no_grad():
            # model 로드
            outputs = model(input_im)

            if len(self.gpus) > 1:
                for output in outputs:
                    parsing = output[0][-1]
                    nums = len(parsing)
                    parsing = interp(parsing).data.cpu().numpy()
                    parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                    parsing_preds[idx:idx + nums, :, :] = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)
                    #idx += nums
            else:
                parsing = outputs[0][-1]
                parsing = interp(parsing).data.cpu().numpy()
                parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                parsing_preds[idx:idx + num_images, :, :] = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)
                #idx += num_images

            parsing_preds = parsing_preds[:1, :, :]

            # json 출력 로직
            label_pred,s,c = parsing_preds[idx,:,:],scales[idx,:],centers[idx,:]

            lbl = transforms_parsing.transform_parsing(label_pred,c,s,org_w,org_h,label_pred.shape,255)
            result = polygon2mask.polygon2json(polygon2mask.mask2polygons(lbl))
            print(result)
        logger.info('완료 !')

## ---------------------------------------Init ----------------------------------------

filepath = "/home/inviz/Desktop/ai/NIA21-Human.v1/input_data/F009/image/01_02_F009_11.jpg"

model = Model()
model.modelInit()

try:
    model.run()
except Exception as e:
    logger.warning(f'model load Test>> {e}')
