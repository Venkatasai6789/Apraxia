"""
ESP32 Hardware Voice Recorder Service
Handles HTTP communication with the ESP32 hardware unit over Wi-Fi REST API.
"""
import os
import re
from typing import Dict, Any, List, Optional
import httpx

ESP32_HOST = os.getenv("ESP32_HOST", "http://192.168.4.1")
DEFAULT_TIMEOUT = 10.0
DOWNLOAD_TIMEOUT = 30.0


class DeviceService:
    def __init__(self, host: Optional[str] = None):
        self.host = (host or ESP32_HOST).rstrip("/")

    @staticmethod
    def sanitize_filename(name: str) -> str:
        """Strip directory traversal and invalid characters from filename."""
        name = name.replace("\\", "/").split("/")[-1].strip()
        # Keep alphanumeric, underscores, hyphens, and dots
        clean = re.sub(r"[^\w\-.]", "_", name)
        if not clean.lower().endswith(".wav"):
            clean += ".wav"
        return clean

    async def get_status(self) -> Dict[str, Any]:
        """Fetch real-time device telemetry and storage status."""
        url = f"{self.host}/api/status"
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    data["connected"] = True
                    return data
                return {
                    "connected": False,
                    "recording": False,
                    "error": f"ESP32 returned HTTP status {resp.status_code}",
                    "total_bytes": 0,
                    "used_bytes": 0,
                    "free_bytes": 0,
                    "file_count": 0,
                }
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
            return {
                "connected": False,
                "recording": False,
                "error": f"ESP32 unreachable at {self.host} ({type(e).__name__})",
                "total_bytes": 0,
                "used_bytes": 0,
                "free_bytes": 0,
                "file_count": 0,
            }
        except Exception as e:
            return {
                "connected": False,
                "recording": False,
                "error": str(e),
                "total_bytes": 0,
                "used_bytes": 0,
                "free_bytes": 0,
                "file_count": 0,
            }

    async def list_recordings(self) -> List[Dict[str, Any]]:
        """List all audio recordings stored on the ESP32 LittleFS flash."""
        url = f"{self.host}/api/recordings"
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return resp.json()
                return []
        except Exception:
            return []

    async def download_recording(self, filename: str) -> Optional[bytes]:
        """Download raw WAV audio binary from ESP32."""
        safe_name = self.sanitize_filename(filename)
        url = f"{self.host}/api/recordings/download"
        try:
            async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT) as client:
                resp = await client.get(url, params={"file": safe_name})
                if resp.status_code == 200:
                    return resp.content
                return None
        except Exception:
            return None

    async def delete_recording(self, filename: str) -> Dict[str, Any]:
        """Delete a recording from ESP32 flash memory."""
        safe_name = self.sanitize_filename(filename)
        url = f"{self.host}/api/recordings/delete"
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                resp = await client.post(url, json={"file": safe_name})
                if resp.status_code == 200:
                    return {"ok": True, "message": "File deleted successfully"}
                error_msg = resp.json().get("error", "Failed to delete file") if resp.headers.get("content-type") == "application/json" else resp.text
                return {"ok": False, "error": error_msg}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def rename_recording(self, old_name: str, new_name: str) -> Dict[str, Any]:
        """Rename a recording file on ESP32 flash memory."""
        safe_old = self.sanitize_filename(old_name)
        safe_new = self.sanitize_filename(new_name)
        url = f"{self.host}/api/recordings/rename"
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                resp = await client.post(url, json={"old_file": safe_old, "new_name": safe_new})
                if resp.status_code == 200:
                    return resp.json()
                error_msg = resp.json().get("error", "Failed to rename file") if "application/json" in resp.headers.get("content-type", "") else resp.text
                return {"ok": False, "error": error_msg}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def start_recording(self) -> Dict[str, Any]:
        """Trigger start of audio recording on ESP32 hardware."""
        url = f"{self.host}/api/record/start"
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                resp = await client.post(url)
                if resp.status_code == 200:
                    return resp.json()
                error_msg = resp.json().get("error", "Failed to start recording") if "application/json" in resp.headers.get("content-type", "") else resp.text
                return {"ok": False, "error": error_msg}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def stop_recording(self) -> Dict[str, Any]:
        """Trigger stop of audio recording on ESP32 hardware."""
        url = f"{self.host}/api/record/stop"
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                resp = await client.post(url)
                if resp.status_code == 200:
                    return resp.json()
                error_msg = resp.json().get("error", "Failed to stop recording") if "application/json" in resp.headers.get("content-type", "") else resp.text
                return {"ok": False, "error": error_msg}
        except Exception as e:
            return {"ok": False, "error": str(e)}
