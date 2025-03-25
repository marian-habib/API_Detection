import numpy as np
import cv2
import pandas as pd
from difflib import get_close_matches
from BBox_Report_Generation import generate_sbbox_report
from Detection_report import generate_detection_report

def convert_to_voc(box, format_type):
    """Convert different bounding box formats to Pascal VOC format (xmin, ymin, xmax, ymax)"""
    if format_type == 'voc':
        return box
    elif format_type == 'coco':
        x, y, w, h = box
        return [x, y, x + w, y + h]
    elif format_type == 'polygon':
        x_coords = box[0::2]
        y_coords = box[1::2]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
    elif format_type == 'rotated':
        xc, yc, w, h, angle = box
        rect = ((xc, yc), (w, h), angle)
        box_pts = cv2.boxPoints(rect)
        x_coords, y_coords = zip(*box_pts)
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
    elif format_type == 'center':
        x_center, y_center, width, height = box
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return [x1, y1, x2, y2]
    else:
        raise ValueError("Unsupported bounding box format")

def map_columns_automatically(df, pred_format, gt_format):
    """Automatically map column names using predefined lists and fuzzy matching."""
 
    # Define possible names for predictions and ground truth based on format
    format_mappings = {
        "voc": {
            "pred": {
                'xmin_pred': ["pred_x_min", "pred_x1", "pred_left", "pred_bbox_xmin"],
                'ymin_pred': ["pred_y_min", "pred_y1", "pred_top", "pred_bbox_ymin"],
                'xmax_pred': ["pred_xmax", "pred_x2", "pred_right", "pred_bbox_xmax"],
                'ymax_pred': ["pred_ymax", "pred_y2", "pred_bottom", "pred_bbox_ymax"],
            },
            "gt": {
                'xmin_gt': ["xmin_gt", "x_min_gt", "x1_gt", "gt_xmin"],
                'ymin_gt': ["ymin_gt", "y_min_gt", "y1_gt", "gt_ymin"],
                'xmax_gt': ["xmax_gt", "x_max_gt", "x2_gt", "gt_xmax"],
                'ymax_gt': ["ymax_gt", "y_max_gt", "y2_gt", "gt_ymax"],
            },
        },
        "coco": {
            "pred": {
                'x_pred': ["pred_x", "pred_x_tl", "coco_pred_x"],
                'y_pred': ["pred_y", "pred_y_tl", "coco_pred_y"],
                'width_pred': ["pred_width", "pred_w", "coco_pred_width"],
                'height_pred': ["pred_height", "pred_h", "coco_pred_height"],
            },
            "gt": {
                'x_gt': ["gt_x", "x_gt", "coco_gt_x"],
                'y_gt': ["gt_y", "y_gt", "coco_gt_y"],
                'width_gt': ["gt_width", "width_gt", "coco_gt_width"],
                'height_gt': ["gt_height", "height_gt", "coco_gt_height"],
            },
        },
        "rotated": {
            "pred": {
                'x_center_pred': ["pred_cx", "pred_x_center"],
                'y_center_pred': ["pred_cy", "pred_y_center"],
                'width_pred': ["pred_width", "pred_w"],
                'height_pred': ["pred_height", "pred_h"],
                'angle_pred': ["pred_angle", "rotated_pred_angle"],
            },
            "gt": {
                'x_center_gt': ["gt_cx", "x_center_gt"],
                'y_center_gt': ["gt_cy", "y_center_gt"],
                'width_gt': ["gt_width", "width_gt"],
                'height_gt': ["gt_height", "height_gt"],
                'angle_gt': ["gt_angle", "rotated_gt_angle"],
            },
        },
        "center": {
            "pred": {
                'x_center_pred': ["pred_cx", "pred_x_center"],
                'y_center_pred': ["pred_cy", "pred_y_center"],
                'width_pred': ["pred_width", "pred_w"],
                'height_pred': ["pred_height", "pred_h"],
            },
            "gt": {
                'x_center_gt': ["gt_cx", "x_center_gt"],
                'y_center_gt': ["gt_cy", "y_center_gt"],
                'width_gt': ["gt_width", "width_gt"],
                'height_gt': ["gt_height", "height_gt"],
            },
        }
    }
 
    # Fetch the correct possible names based on user selection
    possible_names_pred = format_mappings.get(pred_format, {}).get("pred", {})
    possible_names_gt = format_mappings.get(gt_format, {}).get("gt", {})
 
    detected_columns = {}
 
    # Check and map Ground Truth (GT) columns
    for key, possible_variants in possible_names_gt.items():
        matches = get_close_matches(key, df.columns, n=1, cutoff=0.8)
        if matches:
            detected_columns[key] = matches[0]
        else:
            for variant in possible_variants:
                if variant in df.columns:
                    detected_columns[key] = variant
                    break
 
    # Check and map Prediction (Pred) columns
    for key, possible_variants in possible_names_pred.items():
        matches = get_close_matches(key, df.columns, n=1, cutoff=0.8)
        if matches:
            detected_columns[key] = matches[0]
        else:
            for variant in possible_variants:
                if variant in df.columns:
                    detected_columns[key] = variant
                    break
 
 
    return detected_columns


def calculate_iou(gt_box, pred_box):
    """Calculate IoU between two bounding boxes"""
    x_left = max(gt_box[0], pred_box[0])
    y_top = max(gt_box[1], pred_box[1])
    x_right = min(gt_box[2], pred_box[2])
    y_bottom = min(gt_box[3], pred_box[3])

    if x_right < x_left or y_bottom < y_top:
        intersection_area = 0.0
    else:
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    union_area = gt_area + pred_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0

def compute_precision_recall_f1(gt_box, pred_box, epsilon=1e-6):
    """Compute Precision, Recall, and F1 score for bounding box overlap."""
    x_left = max(gt_box[0], pred_box[0])
    y_top = max(gt_box[1], pred_box[1])
    x_right = min(gt_box[2], pred_box[2])
    y_bottom = min(gt_box[3], pred_box[3])

    if x_right < x_left or y_bottom < y_top:
        intersection_area = 0.0
    else:
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    recall = intersection_area / (gt_area + epsilon) if gt_area > 0 else 0.0
    precision = intersection_area / (pred_area + epsilon) if intersection_area > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon) if (precision + recall) > 0 else 0.0

    #recall = intersection_area / (gt_area + epsilon)
    #precision = intersection_area / (pred_area + epsilon)
    #f1_score = 2 * (precision * recall) / (precision + recall + epsilon) if (precision + recall) > 0 else 0.0

    return precision, recall, f1_score

def compute_f2_score(gt_box, pred_box, epsilon=1e-6):
    """Compute F2 Score (emphasizing recall) for bounding box overlap."""
    precision, recall, _ = compute_precision_recall_f1(gt_box, pred_box, epsilon)
    return (5 * precision * recall) / (4 * precision + recall + epsilon) if (precision + recall) > 0 else 0.0
    #return (5 * precision * recall) / (4 * precision + recall + epsilon)

def compute_f1_f2_width(gt_box, pred_box, epsilon=1e-6):
    """Compute F1 and F2 scores based only on width overlap."""
    gt_width = gt_box[2] - gt_box[0]
    pred_width = pred_box[2] - pred_box[0]

    x_left = max(gt_box[0], pred_box[0])
    x_right = min(gt_box[2], pred_box[2])
    intersection_width = max(0, x_right - x_left)

    precision = intersection_width / (pred_width + epsilon)
    recall = intersection_width / (gt_width + epsilon)

    f1_score = 2 * (precision * recall) / (precision + recall + epsilon) if (precision + recall) > 0 else 0.0
    f2_score = (5 * precision * recall) / (4 * precision + recall + epsilon) if (precision + recall) > 0 else 0.0
    #f1_score = 2 * (precision * recall) / (precision + recall + epsilon) if (precision + recall) > 0 else 0.0
    #f2_score = (5 * precision * recall) / (4 * precision + recall + epsilon)

    return f1_score, f2_score

def evaluate_bboxes_from_csv(file_path, output_path, task_type, gt_format, pred_format, error_log_path, choice, report_path):
    """Read CSV, compute IoU and evaluation metrics for each row, and save the results."""
    df = pd.read_csv(file_path)
    column_mapping = map_columns_automatically(df, pred_format, gt_format)
 
    iou_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    f2_scores = []
    
    # Only include width-specific metrics for OCR task type
    include_width_metrics = task_type.lower() == 'ocr'
    
    if include_width_metrics:
        f1_width_scores = []
        f2_width_scores = []
 
    for _, row in df.iterrows():
        # **Handle Prediction Bounding Box**
        if pred_format == "voc":
            pred_box = convert_to_voc([
                row[column_mapping['xmin_pred']], row[column_mapping['ymin_pred']],
                row[column_mapping['xmax_pred']], row[column_mapping['ymax_pred']]
            ], pred_format)
        elif pred_format == "coco":
            pred_box = convert_to_voc([
                row[column_mapping['x_pred']], row[column_mapping['y_pred']],
                row[column_mapping['width_pred']], row[column_mapping['height_pred']]
            ], pred_format)
        elif pred_format == "rotated":
            pred_box = convert_to_voc([
                row[column_mapping['x_center_pred']], row[column_mapping['y_center_pred']],
                row[column_mapping['width_pred']], row[column_mapping['height_pred']],
                row[column_mapping['angle_pred']]
            ], pred_format)
        elif pred_format == "center":
            pred_box = convert_to_voc([
                row[column_mapping['x_center_pred']], row[column_mapping['y_center_pred']],
                row[column_mapping['width_pred']], row[column_mapping['height_pred']]
            ], pred_format)
 
        # **Handle Ground Truth Bounding Box**
        if gt_format == "voc":
            gt_box = convert_to_voc([
                row[column_mapping['xmin_gt']], row[column_mapping['ymin_gt']],
                row[column_mapping['xmax_gt']], row[column_mapping['ymax_gt']]
            ], gt_format)
        elif gt_format == "coco":
            gt_box = convert_to_voc([
                row[column_mapping['x_gt']], row[column_mapping['y_gt']],
                row[column_mapping['width_gt']], row[column_mapping['height_gt']]
            ], gt_format)
        elif gt_format == "rotated":
            gt_box = convert_to_voc([
                row[column_mapping['x_center_gt']], row[column_mapping['y_center_gt']],
                row[column_mapping['width_gt']], row[column_mapping['height_gt']],
                row[column_mapping['angle_gt']]
            ], gt_format)
        elif gt_format == "center":
            gt_box = convert_to_voc([
                row[column_mapping['x_center_gt']], row[column_mapping['y_center_gt']],
                row[column_mapping['width_gt']], row[column_mapping['height_gt']]
            ], gt_format)
 
        # **Compute Metrics**
        iou = calculate_iou(gt_box, pred_box)
        precision, recall, f1 = compute_precision_recall_f1(gt_box, pred_box)
        f2 = compute_f2_score(gt_box, pred_box)
        
        iou_scores.append(iou)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        f2_scores.append(f2)
        
        # Calculate width-specific metrics for OCR task only
        if include_width_metrics:
            f1_width, f2_width = compute_f1_f2_width(gt_box, pred_box)
            f1_width_scores.append(f1_width)
            f2_width_scores.append(f2_width)
 
    df['IoU'] = iou_scores
    df['Precision'] = precision_scores
    df['Recall'] = recall_scores
    df['F1 Score'] = f1_scores
    df['F2 Score'] = f2_scores
    
    if include_width_metrics:
        df['F1 Score Width'] = f1_width_scores
        df['F2 Score Width'] = f2_width_scores
 
    df.to_csv(output_path, index=False)
    
    if task_type.lower() == 'face':
        report_path_f = generate_sbbox_report(output_path, error_log_path, np.mean(iou_scores), np.mean(precision_scores), np.mean(recall_scores), np.mean(f1_scores), np.mean(f2_scores), choice, report_path)
    if include_width_metrics:
        report_path_f = generate_detection_report(output_path)
    response = {
        "status": "success",
        "message": f"Evaluation completed. Results saved to {output_path}",
        "Report Status": f"Report saved to {report_path_f}",
        "metrics": {
            "mean_iou": np.mean(iou_scores),
            "mean_precision": np.mean(precision_scores),
            "mean_recall": np.mean(recall_scores),
            "mean_f1": np.mean(f1_scores),
            "mean_f2": np.mean(f2_scores)
        }
    }
    if include_width_metrics:
        response["metrics"]["mean_f1_width"] = np.mean(f1_width_scores)
        response["metrics"]["mean_f2_width"] = np.mean(f2_width_scores)
    
    return response

def evaluate_bboxes_from_json(predicted_bbox, ground_truth_bbox, task_type, gt_format, pred_format):

    include_width_metrics = task_type.lower() == 'ocr'
    
    pred_bbox = predicted_bbox[0]
    gt_bbox = ground_truth_bbox[0]
    
    if pred_format == "voc":
        pred_box = convert_to_voc([
            pred_bbox.get('xmin_pred'), pred_bbox.get('ymin_pred'),
            pred_bbox.get('xmax_pred'), pred_bbox.get('ymax_pred')
        ], pred_format)
    elif pred_format == "coco":
        pred_box = convert_to_voc([
            pred_bbox.get('x_pred'), pred_bbox.get('y_pred'),
            pred_bbox.get('width_pred'), pred_bbox.get('height_pred')
        ], pred_format)
    elif pred_format == "rotated":
        pred_box = convert_to_voc([
            pred_bbox.get('x_center_pred'), pred_bbox.get('y_center_pred'),
            pred_bbox.get('width_pred'), pred_bbox.get('height_pred'),
            pred_bbox.get('angle_pred')
        ], pred_format)
    elif pred_format == "center":
        pred_box = convert_to_voc([
            pred_bbox.get('x_center_pred'), pred_bbox.get('y_center_pred'),
            pred_bbox.get('width_pred'), pred_bbox.get('height_pred')
        ], pred_format)

    # **Handle Ground Truth Bounding Box**
    if gt_format == "voc":
        gt_box = convert_to_voc([
            gt_bbox.get('xmin_gt'), gt_bbox.get('ymin_gt'),
            gt_bbox.get('xmax_gt'), gt_bbox.get('ymax_gt')
        ], gt_format)
    elif gt_format == "coco":
        gt_box = convert_to_voc([
            gt_bbox.get('x_gt'), gt_bbox.get('y_gt'),
            gt_bbox.get('width_gt'), gt_bbox.get('height_gt')
        ], gt_format)
    elif gt_format == "rotated":
        gt_box = convert_to_voc([
            gt_bbox.get('x_center_gt'), gt_bbox.get('y_center_gt'),
            gt_bbox.get('width_gt'), gt_bbox.get('height_gt'),
            gt_bbox.get('angle_gt')
        ], gt_format)
    elif gt_format == "center":
        gt_box = convert_to_voc([
            gt_bbox.get('x_center_gt'), gt_bbox.get('y_center_gt'),
            gt_bbox.get('width_gt'), gt_bbox.get('height_gt')
        ], gt_format)
    
    iou = calculate_iou(gt_box, pred_box)
    precision, recall, f1 = compute_precision_recall_f1(gt_box, pred_box)
    f2 = compute_f2_score(gt_box, pred_box)

    if include_width_metrics:
        f1_width, f2_width = compute_f1_f2_width(gt_box, pred_box)

    response = {
        "status": "success",
        "message": f"Evaluation completed successfully.",
        "metrics": {
            "IoU":iou,
            "Precision": precision,
            "Recall": recall,
            "F1_score": f1,
            "F2_score": f2
        }
    }
    if include_width_metrics:
        response["metrics"]["F1_width"] = f1_width
        response["metrics"]["F2_width"] = f2_width
    
    return response