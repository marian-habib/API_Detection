from flask import Flask, request, jsonify
from BBox_detection import evaluate_bboxes_from_json,evaluate_bboxes_from_csv
import os

app = Flask(__name__)

@app.route('/data', methods=['POST'])
def df_evaluate():
    try:
        data = request.get_json()
        
        input_file = data.get('input_file')
        output_file = data.get('output_file')
        error_log_path = data.get('error_log_path',None)
        choice = data.get('choice', None)
        report_path = data.get('report_path')
        task_type = data.get('task_type') 
        gt_format = data.get('gt_format') 
        pred_format = data.get('pred_format')
        
        if not os.path.exists(input_file):
            return jsonify({"status": "error", "message": f"Input file {input_file} not found"}), 400
        
        result = evaluate_bboxes_from_csv(input_file, output_file, task_type, gt_format, pred_format, error_log_path, choice, report_path)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        data = request.get_json()

        predicted_bbox = data.get('predicted_bbox', [])
        ground_truth_bbox = data.get('ground_truth_bbox', [])
        task_type = data.get('task_type')
        gt_format = data.get('gt_format')
        pred_format = data.get('pred_format')

        if not predicted_bbox or not ground_truth_bbox:
            return jsonify({"status": "error", "message": "Missing bounding box data"}), 400
        
        result = evaluate_bboxes_from_json(predicted_bbox, ground_truth_bbox, task_type, gt_format, pred_format)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)