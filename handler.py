import runpod

print("handler.py starting...")

def handler(job):
    return {"status": "ok", "message": "hello from handler"}

print("Starting runpod serverless...")
runpod.serverless.start({"handler": handler})
