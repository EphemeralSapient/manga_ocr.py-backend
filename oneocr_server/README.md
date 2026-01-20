# OneOCR Remote Server

Run Windows OneOCR as a network service for non-Windows clients.

## Use Cases

- **Mac/Linux users**: Access Windows OCR from your primary machine
- **Azure Windows VM**: Free tier VM as dedicated OCR server
- **Shared office setup**: One Windows machine serves OCR to multiple clients

## Quick Start (Windows Machine)

### 1. First Run (requires Administrator)

```powershell
# Right-click PowerShell -> Run as Administrator
cd oneocr_server
python server.py
```

This will:
- Copy OneOCR files from Windows Snipping Tool
- Start the HTTP server
- Display the network URL to use

### 2. Subsequent Runs (no admin needed)

```powershell
python server.py
```

### 3. Configure Client

On your main machine, set in `config.json`:

```json
{
  "ocr_method": "oneocr_remote",
  "oneocr_server_url": "http://<WINDOWS_IP>:5050"
}
```

Or run the config wizard and select "OneOCR Remote".

## Server Options

```
python server.py [OPTIONS]

Options:
  --port PORT      Port to run on (default: 5050)
  --host HOST      Host to bind (default: 0.0.0.0)
  --setup-only     Only copy OneOCR files, don't start server
```

## API Endpoints

### Health Check
```
GET /health

Response: {"status": "ok", "service": "oneocr"}
```

### Single Image OCR
```
POST /ocr
Content-Type: application/json

{"image": "<base64_encoded_image>"}

Response: {
  "success": true,
  "results": [
    {"text": "...", "confidence": 0.95, "bbox": [x1, y1, x2, y2]}
  ]
}
```

### Batch OCR
```
POST /ocr/batch
Content-Type: application/json

{"images": ["<base64_1>", "<base64_2>", ...]}

Response: {
  "success": true,
  "count": 2,
  "results": [...]
}
```

## Azure Free Tier Setup

1. Create free Windows VM on Azure (B1s tier)
2. Open port 5050 in Network Security Group
3. RDP into VM, run server setup
4. Use VM's public IP in client config

## Firewall Notes

If clients can't connect:

```powershell
# Allow port 5050 through Windows Firewall
netsh advfirewall firewall add rule name="OneOCR Server" dir=in action=allow protocol=tcp localport=5050
```

## Performance

- OneOCR is ~10x faster than local VLM models
- Network latency adds ~50-200ms per request
- Still significantly faster than VLM alternatives
