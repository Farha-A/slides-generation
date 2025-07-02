bind = "0.0.0.0:8080"
workers = 1
worker_class = "sync"
timeout = 600  # 5 minutes instead of default 30 seconds
keepalive = 2
max_requests = 1000
max_requests_jitter = 100
preload_app = True