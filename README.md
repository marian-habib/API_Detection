# Bounding Boxes Detection API

## Running the Test Framework

1. Open your VS Code project containing all the required files.

2. Run the `app.py` file:
   ```bash
   python app.py
   ```
### If you are going to input json BBox Detection points
3. Open Postman and create a new POST request to `http://localhost:5000/evaluate`

4. Select the `Body` tab, choose `raw` and set the format to `JSON`

5. For the `gt_format` and `pred_format`:
You can choose from those formats:
   voc: [xmin, ymin, xmax, ymax] (Pascal VOC format)
   coco: [x, y, width, height] (COCO format, x&y for the top-left point)
   polygon: [x1, y1, x2, y2, ..., xn, yn] (Polygon format)
   rotated: [x_center, y_center, width, height, angle] (Rotated format)
   center: [x_center, y_center, width, height] (Center format)

6. For face detection tasks:
```bash
{
  "predicted_bbox": [
    {
        "x_pred" : 262,
        "y_pred" : 336,
        "width_pred": 140,
        "height_pred": 140
    }
  ],
  "ground_truth_bbox": [
    {
        "xmin_gt" : 258,
        "ymin_gt" : 304,
        "xmax_gt" : 385,
        "ymax_gt" : 436
    }
  ],
  "task_type": "face",
  "gt_format": "voc",
  "pred_format": "coco"
}
```

7. For OCR tasks:
```bash
{
  "predicted_bbox": [
    {
        "xmin_pred" : 2321,
        "ymin_pred" : 248,
        "xmax_pred": 2699,
        "ymax_pred": 386
    }
  ],
  "ground_truth_bbox": [
    {
        "xmin_gt" : 2334.88,
        "ymin_gt" : 266.01,
        "xmax_gt" : 2679.345,
        "ymax_gt" : 358.05
    }
  ],
  "task_type": "face",
  "gt_format": "voc",
  "pred_format": "voc"
}
```
### If you are going to input a CSV file
3. Open Postman and create a new POST request to `http://localhost:5000/data`

4. Select the `Body` tab, choose `raw` and set the format to `JSON`

5. For the `gt_format` and `pred_format`:
You can choose from those formats:
   voc: [xmin, ymin, xmax, ymax] (Pascal VOC format)
   coco: [x, y, width, height] (COCO format, x&y for the top-left point)
   polygon: [x1, y1, x2, y2, ..., xn, yn] (Polygon format)
   rotated: [x_center, y_center, width, height, angle] (Rotated format)
   center: [x_center, y_center, width, height] (Center format)

6. For face detection tasks:
```bash
{
  "input_file": "input file path",
  "output_file": "output file path",
  "error_log_path": "error file path",
  "report_path": "report_path",
  "choice": "1",
  "task_type": "face",
  "gt_format": "voc",
  "pred_format": "coco"
}
```

7. For OCR tasks:
```bash
{
  "input_file": "input file path",
  "output_file": "output file path",
  "report_path": "report_path",
  "task_type": "ocr",
  "gt_format": "voc",
  "pred_format": "voc"
}
```