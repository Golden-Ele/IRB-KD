#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""python coco_evaluate.py  --annotations path\instances_val2017.json  --predictions path\predictions.json"""


import argparse
import json
import os
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_coco(pred_json, anno_json):
    """
    使用pycocotools评估COCO格式的检测结果
    
    参数:
        pred_json: 预测结果的JSON文件路径
        anno_json: COCO格式的标注文件路径
    
    返回:
        stats: 评估结果统计
    """
    print(f"\n正在评估 COCO 指标，使用 {pred_json} 和 {anno_json}...")
    
    # 检查文件是否存在
    for x in [pred_json, anno_json]:
        assert os.path.isfile(x), f"文件 {x} 不存在"
    
    # 初始化COCO API
    anno = COCO(str(anno_json))  # 初始化标注API
    pred = anno.loadRes(str(pred_json))  # 初始化预测API (必须传递字符串，而非Path对象)
    
    # 进行bbox评估
    eval_bbox = COCOeval(anno, pred, 'bbox')
    eval_bbox.evaluate()
    eval_bbox.accumulate()
    eval_bbox.summarize()


def main():
    parser = argparse.ArgumentParser(description='评估COCO格式的目标检测结果')
    parser.add_argument('--annotations', type=str, required=True, help='COCO格式的标注文件路径')
    parser.add_argument('--predictions', type=str, required=True, help='预测结果的JSON文件路径')
    args = parser.parse_args()
    
    # 确保文件路径存在
    pred_json = Path(args.predictions)
    anno_json = Path(args.annotations)
    
    # 评估并打印结果
    stats = evaluate_coco(pred_json, anno_json)


if __name__ == '__main__':
    main() 
