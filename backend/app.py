import os
import requests
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from flask import stream_with_context

app = Flask(__name__)
UPSTREAM_URL = "http://75.19.1.5:5000/start_predict"
CORS(app)

PC_BACKEND = os.getenv('PC_BACKEND_URL', '').rstrip('/')
if not PC_BACKEND:
    raise RuntimeError("PC_BACKEND_URL environment variable is required")

def stream_events(resp):
    for line in resp.iter_lines():
        if line:
            # Pass the raw “data: …” line straight through
            yield line.decode('utf-8') + '\n'

@app.route('/start_predict', methods=['POST'])
def start_predict():
    try:
        upstream = requests.post(
            UPSTREAM_URL,
            json=request.json,
            stream=True,
            timeout=(5, None),    # 5s connect, unlimited read
        )
        upstream.raise_for_status()
    except requests.RequestException:
        return jsonify({
            "error": "Processing server is currently unavailable. Please try again later."
        }), 503

    # stream raw SSE bytes through to the browser
    return Response(
        stream_with_context(upstream.iter_content(chunk_size=1024)),
        content_type='text/event-stream',
        headers={"Cache-Control": "no-cache"}
    )

@app.route('/predict_updates')
def predict_updates():
    upstream = f"{PC_BACKEND}/predict_updates"
    resp = requests.get(upstream, stream=True)
    return Response(stream_events(resp), mimetype='text/event-stream')

if __name__ == '__main__':
    # Not used in Docker; gunicorn will run it
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
