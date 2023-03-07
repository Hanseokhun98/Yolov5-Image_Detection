import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import time

import torch

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# .exe file path
if getattr(sys, 'frozen', False):
        # we are running in a bundle
        frozen = 'ever so'
        ROOT = sys._MEIPASS
else:
        # we are running in a normal Python environment
        ROOT = os.path.dirname(os.path.abspath(__file__))

ROOT = Path(os.getcwd())

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (LOGGER, Profile, check_img_size, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

file_root = (str)  #detect된 파일경로
file_result = None #detect된 파일의 분류 결과 (v2의 경우 over,under,empty 감지 안되면 normal)

now = time
now_str = now.strftime('%y%m%d') #현재날짜 str로 저장
now_num = int(now_str) #현재날짜 int형으로 변환

#결과 옮길 csv 파일 불러오기
df1 = pd.read_csv('C:/Users/202210/Desktop/Python Project/yolov5_sdent_detect_model/result2.csv')
df1.set_index('date',inplace = True)
# (v3의 경우 normal,error,empty 감지 안되면, no_detect)

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        # imgsz=(640, 640),  # inference size (height, width)
        imgsz=(1280,1280),
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=1)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    file_root = save_path # put file root
                    file_root = file_root.replace("\\","/") # file root setting \ -> /
            
            # len(det) (Meaning: 0 is no detect , 1 is detect error)
            global file_result
            
            #축간 불량 detect 결과
            if (len(det) == 0) : 
                file_result = 'Normal' #정상
            elif (len(det) == 1) :
                if (c == 0) :
                    file_result = 'Normal' #정상
                elif (c == 1) :
                    file_result = 'Error' #불량 
                    df1.at[now_num,'축관 불량 진단 개수'] += 1 #해당 날짜에 축간 불량 용접불량 기록
                    df1.to_csv("C:/Users/202210/Desktop/Python Project/yolov5_sdent_detect_model/result2.csv",mode='w')
                elif (c == 2) :
                    file_result = 'Empty' #시료 없음
            

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

    return file_result

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'cam2_detect.pt', help='model path or triton URL') 
    #가중치: 축간 불량 분석시 'cam2_detect.pt'사용, 용접 불량 분석시 'cam1_detect.pt'사용
    parser.add_argument('--source', type=str, default=ROOT / 'Store/CAM2/', help='file/dir/URL/glob/screen/0(webcam)') 
    #분석할 이미지 저장된 경로 default값에 입력
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280], help='inference size h,w') 
    #이미지 사이즈 1280 추천
    parser.add_argument('--conf-thres', type=float, default=0.45, help='confidence threshold') 
    # (가중치 임계값 수치: 용접 불량 분석시 conf 수치 0.45, 축간 불량 분석시 conf 수치 0.25 로사용)
    parser.add_argument('--project', default=ROOT / 'detect_store', help='save results to project/name') 
    # 감지된 이미지 파일 저장되는 경로 (실시간 용접 불량 detect_store1, 실시간 축간 불량 detect_store2, 사용자 용접 불량 및 축간 불량 이미지 detect_sore3)
    parser.add_argument('--name', default='result2', help='save results to project/name') 
    #위의 project 폴더 하위 폴더로 감지된 파일 담은 폴더 생성
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment') 
    #프로그램 실행시 python detect.py --exist-ok 라고 실행하면 기존 감지된 파일 담은 폴더에 덮어쓰기
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))
    f = open('warehouse2.txt','w') #감지된 결과 텍스트로 저장
    f.write(file_result)
    f.close()
        


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
