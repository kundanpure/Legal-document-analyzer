"""
API Gateway for LegalMind AI Backend
Handles file uploads, validation, and service orchestration
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from service_orchestrator import ServiceOrchestrator
from shared.utils import validate_file, generate_task_id
from shared.gcs_manager import GCSManager
from config import Config

app = Flask(__name__)
CORS(app)

# Initialize components
orchestrator = ServiceOrchestrator()
gcs_manager = GCSManager()
executor = ThreadPoolExecutor(max_workers=4)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'api-gateway',
        'version': '1.0.0'
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'LegalMind AI Gateway is running!',
        'endpoints': {
            'upload': '/analyze-document',
            'status': '/analysis-status/<task_id>',
            'results': '/analysis-results/<task_id>',
            'health': '/health'
        }
    })

@app.route('/analyze-document', methods=['POST'])
def analyze_document():
    """
    Main endpoint for document analysis
    Supports: PDF, DOC, DOCX, TXT, RTF, ODT
    """
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read file content
        file_content = file.read()
        
        # Validate file
        is_valid, error_msg = validate_file(file_content, file.filename)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        # Get additional parameters
        user_query = request.form.get('query', 'Analyze this legal document for key risks and recommendations')
        user_id = request.form.get('user_id', None)
        priority = request.form.get('priority', 'normal')
        
        # Generate task ID
        task_id = generate_task_id()
        
        # Upload to GCS
        try:
            gcs_path = gcs_manager.upload_document(file_content, file.filename, user_id)
        except Exception as e:
            return jsonify({'error': f'Failed to store document: {str(e)}'}), 500
        
        # Start async processing
        executor.submit(
            orchestrator.process_document_async,
            task_id=task_id,
            gcs_path=gcs_path,
            filename=file.filename,
            user_query=user_query,
            user_id=user_id,
            priority=priority
        )
        
        return jsonify({
            'task_id': task_id,
            'status': 'processing',
            'message': 'Document uploaded successfully. Processing started.',
            'gcs_path': gcs_path,
            'estimated_time': '2-5 minutes'
        }), 202
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/analysis-status/<task_id>', methods=['GET'])
def get_analysis_status(task_id):
    """Get analysis status for a task"""
    try:
        status = orchestrator.get_task_status(task_id)
        if status is None:
            return jsonify({'error': 'Task not found'}), 404
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': f'Status check failed: {str(e)}'}), 500

@app.route('/analysis-results/<task_id>', methods=['GET'])
def get_analysis_results(task_id):
    """Get analysis results for a completed task"""
    try:
        results = gcs_manager.load_analysis_results(task_id)
        if results is None:
            return jsonify({'error': 'Results not found'}), 404
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': f'Failed to load results: {str(e)}'}), 500

@app.route('/reanalyze-document/<task_id>', methods=['POST'])
def reanalyze_document(task_id):
    """Reanalyze existing document with new query"""
    try:
        # Get original document info
        original_results = gcs_manager.load_analysis_results(task_id)
        if not original_results:
            return jsonify({'error': 'Original analysis not found'}), 404
        
        # Get new query
        new_query = request.json.get('query')
        if not new_query:
            return jsonify({'error': 'New query is required'}), 400
        
        # Generate new task ID
        new_task_id = generate_task_id()
        
        # Get original GCS path from results
        original_gcs_path = original_results.get('gcs_path')
        if not original_gcs_path:
            return jsonify({'error': 'Original document not found'}), 404
        
        # Start reanalysis
        executor.submit(
            orchestrator.process_document_async,
            task_id=new_task_id,
            gcs_path=original_gcs_path,
            filename=f"reanalysis_{task_id}",
            user_query=new_query,
            user_id=request.json.get('user_id'),
            priority='normal'
        )
        
        return jsonify({
            'task_id': new_task_id,
            'status': 'processing',
            'message': 'Document reanalysis started',
            'original_task_id': task_id
        }), 202
        
    except Exception as e:
        return jsonify({'error': f'Reanalysis failed: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
