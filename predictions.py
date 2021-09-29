import cv2
from pytorchyolo import detect, models
import time
import argparse
import numpy as np
import torch
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Prune Model.")
    parser.add_argument("-p", "--prune", type=bool, default=False, help="Prune model.")
    parser.add_argument("-s", "--sen", type = float, default = 0, help = "Add sensitivty.")
    parser.add_argument("-o", "--operation", type = str, default = "mean", help ="Operation to prune.")
    parser.add_argument("-d", "--display", type = bool, default = False, help = "Display.")
    parser.add_argument("-v", "--video", type = str, default = "sot.mp4", help = "Video.")
    
    args = parser.parse_args()
    model = models.load_model(
    "config/yolov3.cfg", 
    "checkpoints/yolov3_ckpt_299.pth",pruning=True)

#     for name, module in model.named_modules():
#             if type(module).__name__ == "Conv2d":
#                 weights = module.weight.data.cpu().numpy()
#                 if args.operation == "mean":
#                     module_mean = np.mean(weights)
#                     threshold = abs(args.sen * module_mean)
#                 else:
#                     module_std = np.std(weights)
#                     threshold = abs(args.sen*module_std)
#                 new_weights = np.where(abs(weights) < threshold,0,weights)
#                 module.weight.data = torch.from_numpy(new_weights)

    # Load the image as a numpy array
    vid = cv2.VideoCapture(args.video)
    while(True):
        try:
            st_time = time.time()
            ret, frame = vid.read()
            boxes = detect.detect_image(model, frame)
            print(boxes)
            if args.display == True :
                cv2.rectangle(frame, (int(boxes[0][0]), int(boxes[0][1])), (int(boxes[0][2]), int(boxes[0][3])), (0,0,0), 2)
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            fps = 1/(time.time()-st_time)
            print("FPS: ", fps)
        except AttributeError:
            break
      
