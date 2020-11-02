

import argparse
import glob
import os
import shutil
# from argparse import RawTextHelpFormatter
import sys

import _init_paths
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from utils import BBFormat


def getBoundingBoxes(dets,
                     isGT,
                     bbFormat,
                     coordType,
                     allBoundingBoxes=None,
                     allClasses=None,
                     imgSize=(0, 0)):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    if allClasses is None:
        allClasses = []

    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    #nameOfImage = dets[0]

    for i in range(len(dets)):
        dets1 = dets[i]
        nameOfImage = dets1[0]

        if isGT:
            # idClass = int(splitLine[0]) #class
            idClass = 'person'  # class
            x = dets1[1]
            y = dets1[2]
            w = dets1[3]
            h = dets1[4]
            bb = BoundingBox(nameOfImage, idClass, x, y, w, h, coordType, imgSize, BBType.GroundTruth, format=bbFormat)
        else:
            # idClass = int(splitLine[0]) #class
            idClass = 'person'  # class
            confidence = dets1[2]
            x = dets1[3]
            y = dets1[4]
            w = dets1[5]
            h = dets1[6]
            bb = BoundingBox(nameOfImage, idClass, x, y, w, h, coordType, imgSize, BBType.Detected, confidence, format=bbFormat)
        allBoundingBoxes.addBoundingBox(bb)
        if idClass not in allClasses:
            allClasses.append(idClass)

    return allBoundingBoxes, allClasses


gtFormat = BBFormat.XYWH
detFormat = BBFormat.XYWH
gtCoordType = CoordinatesType.Absolute
detCoordType = CoordinatesType.Absolute
imgSize = (800, 800)

def calculate_map(gt, det):
    # Get groundtruth boxes
    allBoundingBoxes, allClasses = getBoundingBoxes(gt, True, gtFormat, gtCoordType, imgSize=imgSize)
    # Get detected boxes
    allBoundingBoxes, allClasses = getBoundingBoxes(det, False, detFormat, detCoordType, allBoundingBoxes, allClasses, imgSize=imgSize)
    allClasses.sort()
    evaluator = Evaluator()
    acc_AP = 0
    validClasses = 0
    
    # Plot Precision x Recall curve
    detections = evaluator.PlotPrecisionRecallCurve(allBoundingBoxes,
        IOUThreshold=0.5,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation,
        )


    # each detection is a class
    for metricsPerClass in detections:
        
        # Get metric values per each class
        cl = metricsPerClass['class']
        ap = metricsPerClass['AP']
        precision = metricsPerClass['precision']
        recall = metricsPerClass['recall']
        totalPositives = metricsPerClass['total positives']
        total_TP = metricsPerClass['total TP']
        total_FP = metricsPerClass['total FP']

        if totalPositives > 0:
            validClasses = validClasses + 1
            acc_AP = acc_AP + ap
            prec = ['%.2f' % p for p in precision]
            rec = ['%.2f' % r for r in recall]
            ap_str = "{0:.2f}%".format(ap * 100)
            # ap_str = "{0:.4f}%".format(ap * 100)

    mAP = acc_AP / validClasses
    mAP_str = "{0:.2f}%".format(mAP * 100)
    
    return mAP
