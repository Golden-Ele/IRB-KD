import argparse
import os
import sys
import warnings

warnings.filterwarnings('ignore')
from pathlib import Path
from ultralytics import YOLO

FILE = Path(__file__).resolve()     # 获取当前脚本绝对路径
ROOT = FILE.parents[0]          # 获取脚本文件的父目录，也就是根目录的路径
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))     # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))     # 转换为相对于当前工作目录的相对路径


def str2bool(str):
    return True if str.lower() == 'true' else False     # .lower()转换为小写字母


def transformer_opt(opt):
    opt = vars(opt)
    if opt['unamp']:
        opt['amp'] = False
    else:
        opt['amp'] = True
    del opt['yaml']
    del opt['weight']
    del opt['teacher_weight']
    del opt['info']
    del opt['unamp']
    return opt


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default='ultralytics/models/v8/yolov8n.yaml', help='model.yaml path')
    # 要蒸馏的学生模型
    parser.add_argument('--weight', type=str, default='yolov8n.pt', help='pretrained model path')
    parser.add_argument('--teacher-weight', type=str, required=True, help='trained teacher checkpoint path')
    parser.add_argument('--cfg', type=str, default='hyp.yaml', help='hyperparameters path')
    parser.add_argument('--data', type=str, default='ultralytics/datasets/UAVDT.yaml', help='data yaml path')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--patience', type=int, default=50, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--unamp', action='store_true', default=False,
                        help='Unuse Automatic Mixed Precision (AMP) training')
    parser.add_argument('--batch', type=int, default=256, help='number of images per batch (-1 for AutoBatch)')
    parser.add_argument('--imgsz', type=int, default=640, help='size of input images as integer')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--device', type=str, default='0,1,2,3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', type=str, default='runs/train-distillation-UAVDT', help='save to project/name')
    parser.add_argument('--name', type=str, default='mgd', help='save to project/name')      
    parser.add_argument('--resume', type=str, default='', help='resume training from last checkpoint')
    parser.add_argument('--optimizer', type=str,
                        choices=['SGD', 'Adam', 'Adamax', 'NAdam', 'RAdam', 'AdamW', 'RMSProp', 'auto'],                            
                        default='SGD',
                        help='optimizer (auto -> ultralytics/yolo/engine/trainer.py in build_optimizer funciton.)')             
    parser.add_argument('--close_mosaic', type=int, default=0,
                        help='(int) disable mosaic augmentation for final epochs')
    parser.add_argument('--info', action="store_true", default=False, help='model info verbose')
    parser.add_argument('--save', type=str2bool, default='True', help='save train checkpoints and DET results')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--seed', type=int, default=77, help='Global training seed')
    parser.add_argument('--deterministic', action="store_true", default=True,
                        help='whether to enable deterministic mode')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--cos-lr', action='store_true', default=False, help='cosine LR scheduler')
    parser.add_argument('--fraction', type=float, default=1.0,
                        help='dataset fraction to train on (default is 1.0, all images in train set)')
    parser.add_argument('--profile', action='store_true', default=False,
                        help='profile ONNX and TensorRT speeds during training for loggers')

    return parser.parse_known_args()[0]


class YOLOV8(YOLO):
    """
    yaml:model.yaml path
    weight:pretrained model path
    """

    def __init__(self, yaml='', weight='', task=None) -> None:
        super().__init__(yaml, task)
        if weight:
            self.load(weight)


if __name__ == '__main__':
    opt = parse_opt()
    model_t = YOLO(opt.teacher_weight)     # 选择教师模型   
    model_t.model.model[-1].set_Distillation = True           # 是否设置为蒸馏模型
    model_s = YOLOV8(yaml=opt.yaml, weight=opt.weight, task='detect')
    model_s.train(**transformer_opt(opt), Distillation=model_t.model)
