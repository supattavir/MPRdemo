import options.options as option
import utils.util as util
from models import create_model
import torch
import cv2
import numpy as np
import os
import time

opt = option.parse('nESRGANplus2.json', is_train=False)
util.mkdirs((path for key, path in opt['path'].items() if not key == 'pretrain_model_G'))
opt = option.dict_to_nonedict(opt)
model = create_model(opt)
device = torch.device('mps')
model.device = device
model.netG.to(device)
model.netG.eval()

cap = cv2.VideoCapture('../vdo/Twitter.mp4')
savepath = '../vdo/sr'
iframe = 0
t1 = time.time()
while True:
    iframe += 1
    _, frame = cap.read()
    if frame is None:
        break
    if iframe < 400:
        continue
    if frame.max() > 1:
        frame = frame / 255.
    frame = frame[None, :, :, ::-1].copy()
    im = torch.FloatTensor(frame).to(model.device)
    im = im.permute(0, 3, 1, 2)

    with torch.no_grad():
        z = model.netG(im)
        if device.type == 'cpu':
            z = z[0].permute(1, 2, 0).detach().numpy()
        else:
            z = z[0].permute(1, 2, 0).detach().cpu().numpy()
        print(iframe, frame.shape[1:], z.shape, time.time()-t1)
        cv2.imwrite(os.path.join(savepath, str(iframe) + '.png'), np.uint8(z[:, :, ::-1] * 255))

