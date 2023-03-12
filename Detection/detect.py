import argparse
import time
from pathlib import Path
import os

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size,
    check_requirements,
    check_imshow,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    strip_optimizer,
    set_logging,
    increment_path,
)
from utils.plots import plot_one_box
from utils.torch_utils import (
    select_device,
    load_classifier,
    time_synchronized,
    TracedModel,
)


def rescaleFrame(frame, scale=0.5):
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)
    dimesnions = (width, height)
    return cv2.resize(frame, dimesnions, interpolation=cv2.INTER_AREA)


def detect(source, weights, device, img_size, iou_thres, conf_thres):
    # webcam = source.isnumeric()

    # init
    set_logging()
    device = select_device(device)
    half = device.type != "cpu"

    # load model
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(img_size, s=stride)

    if half:
        model.half()

    # if webcam:
    #     view_img = check_imshow()
    #     cudnn.benchmark = True
    #     dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    names = model.module.name if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if device.type != "cpu":
        model(
            torch.zeros(1, 3, imgsz, imgsz).to(
                device).type_as(next(model.parameters()))
        )

    old_img_w = old_img_h = img_size
    old_img_b = 1

    t0 = time.perf_counter()

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        if device.type != "cpu" and (
            old_img_b != img.shape[0]
            or old_img_h != img.shape[2]
            or old_img_w != img.shape[3]
        ):
            old_img_b = img_size[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[4]

        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img)[0]
        t2 = time_synchronized()

        pred = non_max_suppression(pred, conf_thres, iou_thres)
        t3 = time_synchronized()

        for i, det in enumerate(pred):
            p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)

            p = Path(p)

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]} {'s' * (n > 1)},"

                for *xyxy, conf, cls in reversed(det):
                    label = f"{names[int(cls)]} {conf:.2f}"
                    plot_one_box(
                        xyxy, im0, label=label, color=colors[int(
                            cls)], line_thickness=5
                    )

        cv2.imwrite("detected_img.jpg", rescaleFrame(im0))

    print(f"Done . ({time.perf_counter() - t0:.3f}s)")


def run(src, imsize, iou, conf):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(device)

    # weights for person and white-board detection

    with torch.no_grad():
        detect(
            src,
            "best1.pt",
            device,
            img_size=imsize,
            iou_thres=iou,
            conf_thres=conf,
        )
