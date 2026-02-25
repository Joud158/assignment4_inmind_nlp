# IMDB Sentiment REST API — FastAPI

This project deploys our **best IMDB sentiment model** as a REST API using **FastAPI**.  

**Prepared for InMind by:**  
**Joud Senan** and **Leticia Mallat**

---

pip install -r requirements.txt   
$env:BEST_RUN_NAME="bilstm_tfidf_h64_lr0.001_bs64_do0.3"   
uvicorn main:app --reload --port 8000   
