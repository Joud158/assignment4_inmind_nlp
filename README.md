pip install -r requirements.txt
$env:BEST_RUN_NAME="bilstm_tfidf_h64_lr0.001_bs64_do0.3"
uvicorn main:app --reload --port 8000