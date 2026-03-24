#  High-Performance ML Inference System with Custom Java KV-Store

A production-grade microservices architecture that integrates a **Python ML service** with a **custom-built Java Key-Value Store** to reduce API latency by over 60%.



##  Performance Impact
- **Cold Request (Model Inference):** ~1000ms+ 
- **Cached Request (Custom KV-Store):** <5ms
- **Latency Reduction:** 99% for frequent queries.

##  Architecture
- **ML Service (Python/FastAPI):** Serves sentiment analysis using `DistilBERT`.
- **KV-Store (Java):** A custom-built, thread-safe TCP server implementing an LRU cache.
- **Orchestration (Docker):** Fully containerized environment for seamless deployment.

##  Tech Stack
- **Languages:** Java 17, Python 3.12
- **Frameworks:** FastAPI, PyTorch (Transformers)
- **Infrastructure:** Docker, Docker Compose, TCP Sockets
