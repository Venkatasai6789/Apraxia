#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <LittleFS.h>
#include <WiFi.h>
#include <WebServer.h>
#include "driver/i2s.h"

// ============================================================================
// OLED DISPLAY CONFIGURATION
// ============================================================================
#define OLED_SDA       21
#define OLED_SCL       22
#define SCREEN_WIDTH   128
#define SCREEN_HEIGHT  64
#define OLED_RESET     -1
#define OLED_ADDRESS   0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// ============================================================================
// INMP441 I2S MICROPHONE CONFIGURATION
// ============================================================================
#define I2S_BCLK  26
#define I2S_WS    25
#define I2S_DIN   33
#define I2S_PORT  I2S_NUM_0

// ============================================================================
// AUDIO RECORDING PARAMETERS
// ============================================================================
#define SAMPLE_RATE          16000
#define MAX_RECORD_SECONDS   30
#define RECORD_DIR           "/recordings"
#define BUFFER_SAMPLES       256

// 16 kHz * 30 seconds * 2 bytes/sample (16-bit mono) = 960,000 bytes
static const uint32_t MAX_AUDIO_BYTES =
  (uint32_t)SAMPLE_RATE * MAX_RECORD_SECONDS * 2UL;

File audioFile;
bool isRecording = false;
uint32_t audioBytes = 0;
uint32_t recordingStartMs = 0;
uint32_t lastDisplayUpdateMs = 0;
String currentFile = "";

int32_t rawSamples[BUFFER_SAMPLES];
int16_t pcmSamples[BUFFER_SAMPLES];

// ============================================================================
// WI-FI & WEB SERVER CONFIGURATION
// ============================================================================
const char* AP_SSID = "ESP32_Voice_Recorder";
const char* AP_PASSWORD = "Voice@1234"; // WPA2 requires >= 8 characters
WebServer server(80);

// ============================================================================
// DISPLAY HELPERS
// ============================================================================
void showMessage(const String &line1, const String &line2) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 14);
  display.println(line1);
  display.setCursor(0, 34);
  display.println(line2);
  display.display();
}

void updateCountdown() {
  uint32_t elapsedMs = millis() - recordingStartMs;
  uint32_t elapsedSeconds = elapsedMs / 1000UL;
  int remaining = MAX_RECORD_SECONDS - (int)elapsedSeconds;
  if (remaining < 0) remaining = 0;

  char timerText[8];
  snprintf(timerText, sizeof(timerText), "00:%02d", remaining);

  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(25, 5);
  display.println("RECORDING...");
  display.setTextSize(2);
  display.setCursor(28, 25);
  display.println(timerText);
  display.setTextSize(1);
  display.setCursor(30, 52);
  display.println("Speak now");
  display.display();
}

// ============================================================================
// WAV HEADER (Standard 44-byte Little-Endian PCM Header)
// ============================================================================
bool writeWavHeader(File &file, uint32_t dataSize) {
  uint8_t header[44] = {0};

  const uint32_t riffChunkSize = dataSize + 36UL;
  const uint32_t fmtChunkSize = 16;
  const uint32_t sampleRate = SAMPLE_RATE;
  const uint32_t byteRate = SAMPLE_RATE * 2UL; // mono, 16-bit = 32000 B/s
  const uint16_t audioFormat = 1;              // PCM
  const uint16_t channels = 1;
  const uint16_t blockAlign = 2;
  const uint16_t bitsPerSample = 16;

  memcpy(header + 0,  "RIFF", 4);
  memcpy(header + 4,  &riffChunkSize, 4);
  memcpy(header + 8,  "WAVE", 4);
  memcpy(header + 12, "fmt ", 4);
  memcpy(header + 16, &fmtChunkSize, 4);
  memcpy(header + 20, &audioFormat, 2);
  memcpy(header + 22, &channels, 2);
  memcpy(header + 24, &sampleRate, 4);
  memcpy(header + 28, &byteRate, 4);
  memcpy(header + 32, &blockAlign, 2);
  memcpy(header + 34, &bitsPerSample, 2);
  memcpy(header + 36, "data", 4);
  memcpy(header + 40, &dataSize, 4);

  if (!file.seek(0)) return false;
  return file.write(header, sizeof(header)) == sizeof(header);
}

// ============================================================================
// SAFE FILENAME & UTILITIES
// ============================================================================
bool isSafeFileName(const String &name) {
  if (name.length() < 5 || name.length() > 64) return false;
  if (!name.endsWith(".wav") && !name.endsWith(".WAV")) return false;

  int baseLen = name.length() - 4;
  if (baseLen < 1) return false;

  // First character must be alphanumeric
  if (!isalnum(name[0])) return false;

  for (int i = 0; i < baseLen; ++i) {
    char c = name[i];
    if (!isalnum(c) && c != '_' && c != '-') {
      return false; // Rejects spaces, dots, slashes, null bytes
    }
  }
  return true;
}

String cleanFileNameArg(String raw) {
  raw.trim();
  int slash = raw.lastIndexOf('/');
  if (slash >= 0) raw = raw.substring(slash + 1);
  return raw;
}

String generateFileName() {
  for (uint16_t i = 1; i <= 9999; ++i) {
    char path[40];
    snprintf(path, sizeof(path), RECORD_DIR "/REC%04u.wav", i);
    if (!LittleFS.exists(path)) return String(path);
  }
  return String("");
}

// ============================================================================
// I2S SETUP
// ============================================================================
bool setupI2S() {
  i2s_config_t config = {};
  config.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
  config.sample_rate = SAMPLE_RATE;
  config.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
  config.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  config.communication_format = I2S_COMM_FORMAT_I2S;
  config.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  config.dma_buf_count = 8;
  config.dma_buf_len = BUFFER_SAMPLES;
  config.use_apll = false;
  config.tx_desc_auto_clear = false;
  config.fixed_mclk = 0;

  esp_err_t err = i2s_driver_install(I2S_PORT, &config, 0, nullptr);
  if (err != ESP_OK) {
    Serial.printf("I2S driver install failed: %d\n", (int)err);
    return false;
  }

  i2s_pin_config_t pins = {};
  pins.bck_io_num = I2S_BCLK;
  pins.ws_io_num = I2S_WS;
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  pins.data_in_num = I2S_DIN;
  pins.mck_io_num = I2S_PIN_NO_CHANGE;

  err = i2s_set_pin(I2S_PORT, &pins);
  if (err != ESP_OK) {
    Serial.printf("I2S pin setup failed: %d\n", (int)err);
    i2s_driver_uninstall(I2S_PORT);
    return false;
  }

  i2s_zero_dma_buffer(I2S_PORT);
  return true;
}

// ============================================================================
// RECORDING CONTROL
// ============================================================================
void stopRecording(const char *reason) {
  if (!isRecording) {
    Serial.println("No active recording.");
    return;
  }

  isRecording = false;

  bool headerOK = writeWavHeader(audioFile, audioBytes);
  audioFile.flush();
  audioFile.close();

  Serial.println();
  Serial.println(reason);
  Serial.print("Saved file: ");
  Serial.println(currentFile);
  Serial.print("Audio data: ");
  Serial.print(audioBytes);
  Serial.println(" bytes");
  Serial.print("Approx. duration: ");
  Serial.print((float)audioBytes / (SAMPLE_RATE * 2.0f), 2);
  Serial.println(" seconds");

  if (!headerOK) {
    Serial.println("WARNING: WAV header could not be finalized.");
    showMessage("SAVE WARNING", "Check serial log");
  } else {
    showMessage("RECORDING SAVED", "WAV file ready");
  }

  delay(1000);
  showMessage("VOICE RECORDER", "READY");
}

bool startRecording() {
  if (isRecording) {
    Serial.println("Already recording.");
    return false;
  }

  size_t total = LittleFS.totalBytes();
  size_t used = LittleFS.usedBytes();
  size_t freeBytes = (total > used) ? (total - used) : 0;

  if (freeBytes < (MAX_AUDIO_BYTES + 4096UL)) {
    Serial.println("Not enough free LittleFS space for recording.");
    Serial.printf("Free: %u bytes; required: %u bytes\n",
                  (unsigned)freeBytes,
                  (unsigned)(MAX_AUDIO_BYTES + 4096UL));
    showMessage("STORAGE FULL", "Delete old files");
    return false;
  }

  currentFile = generateFileName();
  if (currentFile.length() == 0) {
    Serial.println("Could not generate a recording filename.");
    return false;
  }

  audioFile = LittleFS.open(currentFile, "w");
  if (!audioFile) {
    Serial.println("Could not create recording file.");
    showMessage("FILE ERROR", "Check LittleFS");
    return false;
  }

  uint8_t blankHeader[44] = {0};
  if (audioFile.write(blankHeader, sizeof(blankHeader)) != sizeof(blankHeader)) {
    Serial.println("Could not reserve WAV header.");
    audioFile.close();
    LittleFS.remove(currentFile);
    return false;
  }

  audioBytes = 0;
  i2s_zero_dma_buffer(I2S_PORT);

  recordingStartMs = millis();
  lastDisplayUpdateMs = 0;
  isRecording = true;

  Serial.println();
  Serial.println("RECORDING STARTED");
  Serial.print("File: ");
  Serial.println(currentFile);

  updateCountdown();
  return true;
}

void recordAudio() {
  if (!isRecording) return;

  if (millis() - recordingStartMs >= MAX_RECORD_SECONDS * 1000UL) {
    stopRecording("Maximum recording time reached.");
    return;
  }

  size_t bytesRead = 0;
  esp_err_t result = i2s_read(
    I2S_PORT,
    rawSamples,
    sizeof(rawSamples),
    &bytesRead,
    pdMS_TO_TICKS(20)
  );

  if (result != ESP_OK || bytesRead == 0) return;

  int sampleCount = bytesRead / sizeof(int32_t);
  if (sampleCount > BUFFER_SAMPLES) sampleCount = BUFFER_SAMPLES;

  uint32_t recordedSamples = audioBytes / 2UL;
  uint32_t maxSamples = (uint32_t)SAMPLE_RATE * MAX_RECORD_SECONDS;
  uint32_t remainingSamples =
    (recordedSamples < maxSamples) ? (maxSamples - recordedSamples) : 0;

  if ((uint32_t)sampleCount > remainingSamples) {
    sampleCount = (int)remainingSamples;
  }

  for (int i = 0; i < sampleCount; ++i) {
    // INMP441 24-bit sample is left-aligned in the 32-bit I2S slot.
    pcmSamples[i] = (int16_t)(rawSamples[i] >> 16);
  }

  if (sampleCount > 0) {
    size_t bytesToWrite = (size_t)sampleCount * sizeof(int16_t);
    size_t written = audioFile.write((const uint8_t *)pcmSamples, bytesToWrite);
    audioBytes += written;

    if (written != bytesToWrite) {
      stopRecording("Storage write error; recording stopped.");
      return;
    }
  }

  if (millis() - lastDisplayUpdateMs >= 200) {
    lastDisplayUpdateMs = millis();
    updateCountdown();
  }

  if (audioBytes >= MAX_AUDIO_BYTES ||
      millis() - recordingStartMs >= MAX_RECORD_SECONDS * 1000UL) {
    stopRecording("Maximum recording time reached.");
  }
}

// ============================================================================
// CORS HELPERS
// ============================================================================
void setCorsHeaders() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.sendHeader("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
  server.sendHeader("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With");
}

void handleOptions() {
  setCorsHeaders();
  server.send(204);
}

// ============================================================================
// REST API ENDPOINTS
// ============================================================================

// GET /api/status
void handleApiStatus() {
  setCorsHeaders();
  size_t total = LittleFS.totalBytes();
  size_t used = LittleFS.usedBytes();
  size_t freeBytes = (total > used) ? (total - used) : 0;

  int count = 0;
  File dir = LittleFS.open(RECORD_DIR);
  if (dir && dir.isDirectory()) {
    File entry = dir.openNextFile();
    while (entry) {
      String name = cleanFileNameArg(entry.name());
      if (isSafeFileName(name)) count++;
      entry.close();
      entry = dir.openNextFile();
    }
    dir.close();
  }

  String json = "{";
  json += "\"connected\":true,";
  json += "\"recording\":" + String(isRecording ? "true" : "false") + ",";
  json += "\"current_file\":\"" + currentFile + "\",";
  json += "\"total_bytes\":" + String((unsigned long)total) + ",";
  json += "\"used_bytes\":" + String((unsigned long)used) + ",";
  json += "\"free_bytes\":" + String((unsigned long)freeBytes) + ",";
  json += "\"file_count\":" + String(count) + ",";
  json += "\"max_record_seconds\":" + String(MAX_RECORD_SECONDS) + ",";
  json += "\"sample_rate\":" + String(SAMPLE_RATE) + ",";
  json += "\"uptime_ms\":" + String((unsigned long)millis()) + ",";
  json += "\"wifi_clients\":" + String(WiFi.softAPgetStationNum());
  json += "}";

  server.send(200, "application/json", json);
}

// GET /api/recordings
void handleApiRecordings() {
  setCorsHeaders();
  String json = "[";
  bool first = true;

  File dir = LittleFS.open(RECORD_DIR);
  if (dir && dir.isDirectory()) {
    File entry = dir.openNextFile();
    while (entry) {
      String name = cleanFileNameArg(entry.name());
      if (isSafeFileName(name)) {
        size_t fileSize = entry.size();
        float duration = 0.0f;
        if (fileSize > 44) {
          duration = (float)(fileSize - 44) / (SAMPLE_RATE * 2.0f);
        }

        if (!first) json += ",";
        first = false;

        json += "{";
        json += "\"name\":\"" + name + "\",";
        json += "\"size\":" + String((unsigned long)fileSize) + ",";
        json += "\"duration_sec\":" + String(duration, 2);
        json += "}";
      }
      entry.close();
      entry = dir.openNextFile();
    }
    dir.close();
  }

  json += "]";
  server.send(200, "application/json", json);
}

// GET /api/recordings/download?file=REC0001.wav
void handleApiDownload() {
  setCorsHeaders();
  if (isRecording) {
    server.send(409, "application/json", "{\"ok\":false,\"error\":\"Recording active\"}");
    return;
  }

  if (!server.hasArg("file")) {
    server.send(400, "application/json", "{\"ok\":false,\"error\":\"Missing file argument\"}");
    return;
  }

  String name = cleanFileNameArg(server.arg("file"));
  if (!isSafeFileName(name)) {
    server.send(400, "application/json", "{\"ok\":false,\"error\":\"Invalid filename\"}");
    return;
  }

  String path = String(RECORD_DIR) + "/" + name;
  File file = LittleFS.open(path, "r");
  if (!file || file.isDirectory()) {
    if (file) file.close();
    server.send(404, "application/json", "{\"ok\":false,\"error\":\"File not found\"}");
    return;
  }

  server.sendHeader("Content-Disposition", "attachment; filename=\"" + name + "\"");
  server.sendHeader("Cache-Control", "no-cache");
  server.streamFile(file, "audio/wav");
  file.close();
}

// POST /api/recordings/delete
void handleApiDelete() {
  setCorsHeaders();
  if (isRecording) {
    server.send(409, "application/json", "{\"ok\":false,\"error\":\"Cannot delete while recording\"}");
    return;
  }

  String name = "";
  if (server.hasArg("file")) {
    name = server.arg("file");
  } else if (server.hasArg("plain")) {
    // Check if JSON body
    String body = server.arg("plain");
    int idx = body.indexOf("\"file\":");
    if (idx >= 0) {
      int start = body.indexOf('"', idx + 7);
      int end = body.indexOf('"', start + 1);
      if (start >= 0 && end > start) name = body.substring(start + 1, end);
    }
  }

  name = cleanFileNameArg(name);
  if (!isSafeFileName(name)) {
    server.send(400, "application/json", "{\"ok\":false,\"error\":\"Invalid filename\"}");
    return;
  }

  String path = String(RECORD_DIR) + "/" + name;
  if (!LittleFS.exists(path)) {
    server.send(404, "application/json", "{\"ok\":false,\"error\":\"File not found\"}");
    return;
  }

  if (!LittleFS.remove(path)) {
    server.send(500, "application/json", "{\"ok\":false,\"error\":\"Failed to delete file\"}");
    return;
  }

  server.send(200, "application/json", "{\"ok\":true,\"message\":\"File deleted\"}");
}

// POST /api/recordings/rename
void handleApiRename() {
  setCorsHeaders();
  if (isRecording) {
    server.send(409, "application/json", "{\"ok\":false,\"error\":\"Cannot rename while recording\"}");
    return;
  }

  String oldName = "";
  String newName = "";

  if (server.hasArg("old_file")) oldName = server.arg("old_file");
  else if (server.hasArg("file")) oldName = server.arg("file");

  if (server.hasArg("new_name")) newName = server.arg("new_name");
  else if (server.hasArg("new_file")) newName = server.arg("new_file");

  // Fallback to JSON body if not in query/form
  if ((oldName.length() == 0 || newName.length() == 0) && server.hasArg("plain")) {
    String body = server.arg("plain");
    int idxOld = body.indexOf("\"old_name\":");
    if (idxOld < 0) idxOld = body.indexOf("\"file\":");
    if (idxOld >= 0) {
      int s = body.indexOf('"', idxOld + 7);
      int e = body.indexOf('"', s + 1);
      if (s >= 0 && e > s) oldName = body.substring(s + 1, e);
    }
    int idxNew = body.indexOf("\"new_name\":");
    if (idxNew >= 0) {
      int s = body.indexOf('"', idxNew + 11);
      int e = body.indexOf('"', s + 1);
      if (s >= 0 && e > s) newName = body.substring(s + 1, e);
    }
  }

  oldName = cleanFileNameArg(oldName);
  newName = cleanFileNameArg(newName);

  if (!newName.endsWith(".wav") && !newName.endsWith(".WAV")) {
    newName += ".wav";
  }

  if (!isSafeFileName(oldName) || !isSafeFileName(newName)) {
    server.send(400, "application/json", "{\"ok\":false,\"error\":\"Invalid filename format\"}");
    return;
  }

  String oldPath = String(RECORD_DIR) + "/" + oldName;
  String newPath = String(RECORD_DIR) + "/" + newName;

  if (!LittleFS.exists(oldPath)) {
    server.send(404, "application/json", "{\"ok\":false,\"error\":\"Source file not found\"}");
    return;
  }

  if (LittleFS.exists(newPath) && !oldName.equalsIgnoreCase(newName)) {
    server.send(409, "application/json", "{\"ok\":false,\"error\":\"Target filename already exists\"}");
    return;
  }

  if (!LittleFS.rename(oldPath, newPath)) {
    server.send(500, "application/json", "{\"ok\":false,\"error\":\"Failed to rename file\"}");
    return;
  }

  String json = "{\"ok\":true,\"old_name\":\"" + oldName + "\",\"new_name\":\"" + newName + "\"}";
  server.send(200, "application/json", json);
}

// POST /api/record/start
void handleApiRecordStart() {
  setCorsHeaders();
  if (isRecording) {
    server.send(409, "application/json", "{\"ok\":false,\"error\":\"Already recording\"}");
    return;
  }

  if (!startRecording()) {
    server.send(500, "application/json", "{\"ok\":false,\"error\":\"Failed to start recording (check storage)\"}");
    return;
  }

  String json = "{\"ok\":true,\"file\":\"" + currentFile + "\",\"max_seconds\":" + String(MAX_RECORD_SECONDS) + "}";
  server.send(200, "application/json", json);
}

// POST /api/record/stop
void handleApiRecordStop() {
  setCorsHeaders();
  if (!isRecording) {
    server.send(400, "application/json", "{\"ok\":false,\"error\":\"No active recording\"}");
    return;
  }

  String saved = currentFile;
  uint32_t bytes = audioBytes;
  stopRecording("Stopped via REST API.");

  String json = "{\"ok\":true,\"file\":\"" + saved + "\",\"bytes\":" + String((unsigned long)bytes) + "}";
  server.send(200, "application/json", json);
}

// ============================================================================
// HTML WEB INTERFACE (PRESERVED EXACTLY AS TESTED)
// ============================================================================
void handleWebRoot() {
  String page;
  page.reserve(3800);

  page += F("<!DOCTYPE html><html><head><meta name='viewport' content='width=device-width,initial-scale=1'>");
  page += F("<title>ESP32 Voice Recorder</title><style>");
  page += F("body{font-family:Arial,sans-serif;background:#f3f5f7;margin:0;padding:18px;color:#18212b}");
  page += F(".wrap{max-width:680px;margin:auto}.card{background:white;padding:18px;border-radius:14px;margin:14px 0;box-shadow:0 2px 10px #0001}");
  page += F("h2{margin-top:0}.muted{color:#667085;font-size:14px}.file{border-top:1px solid #e6e8eb;padding:14px 0}");
  page += F("a.btn,button{display:inline-block;background:#1769e0;color:white;padding:10px 14px;border:0;border-radius:8px;text-decoration:none;font-size:14px;margin:4px 4px 4px 0;cursor:pointer}");
  page += F("a.del,button.del{background:#c73535}.empty{padding:16px;background:#f8fafc;border-radius:8px}");
  page += F("</style></head><body><div class='wrap'>");
  page += F("<div class='card'><h2>ESP32 Voice Recorder</h2>");
  page += F("<div class='muted'>Recordings stored in ESP32 internal flash (LittleFS)</div>");
  page += F("<p><b>Recorder status:</b> ");
  page += (isRecording ? "Recording in progress" : "Ready");
  page += F("</p><p><b>Wi-Fi:</b> ESP32_Voice_Recorder</p>");
  page += F("<p><b>Address:</b> http://192.168.4.1</p></div>");

  page += F("<div class='card'><h3>Saved WAV recordings</h3>");

  if (isRecording) {
    page += F("<p class='empty'>A recording is currently in progress. Stop or wait for it to finish before downloading.</p>");
  }

  File dir = LittleFS.open(RECORD_DIR);
  bool found = false;

  if (dir && dir.isDirectory()) {
    File entry = dir.openNextFile();
    while (entry) {
      String name = cleanFileNameArg(entry.name());
      size_t fileSize = entry.size();

      if (isSafeFileName(name)) {
        found = true;
        page += F("<div class='file'><b>");
        page += name;
        page += F("</b><div class='muted'>");
        page += String((unsigned long)fileSize);
        page += F(" bytes</div>");

        if (!isRecording) {
          page += F("<a class='btn' href='/download?file=");
          page += name;
          page += F("'>Download WAV</a>");
          page += F("<form style='display:inline' method='POST' action='/delete' onsubmit='return confirm(\"Delete this recording?\")'>");
          page += F("<input type='hidden' name='file' value='");
          page += name;
          page += F("'><button class='del' type='submit'>Delete</button></form>");
        } else {
          page += F("<span class='muted'>Available after recording stops</span>");
        }
        page += F("</div>");
      }
      entry.close();
      entry = dir.openNextFile();
    }
    dir.close();
  }

  if (!found) {
    page += F("<p class='empty'>No saved WAV recordings found. Use the recorder to create one.</p>");
  }

  page += F("</div><div class='card'><h3>Storage</h3><p>Total: ");
  page += String((unsigned long)LittleFS.totalBytes());
  page += F(" bytes</p><p>Used: ");
  page += String((unsigned long)LittleFS.usedBytes());
  page += F(" bytes</p><p>Free: ");
  size_t total = LittleFS.totalBytes();
  size_t used = LittleFS.usedBytes();
  page += String((unsigned long)(total > used ? total - used : 0));
  page += F(" bytes</p><a class='btn' href='/'>Refresh list</a></div>");
  page += F("</div></body></html>");

  server.send(200, "text/html; charset=utf-8", page);
}

void handleWebDownload() {
  handleApiDownload();
}

void handleWebDelete() {
  if (isRecording) {
    server.send(409, "text/plain", "Cannot delete files while recording.");
    return;
  }

  if (!server.hasArg("file")) {
    server.send(400, "text/plain", "Missing file name.");
    return;
  }

  String name = cleanFileNameArg(server.arg("file"));
  if (!isSafeFileName(name)) {
    server.send(400, "text/plain", "Invalid file name.");
    return;
  }

  String path = String(RECORD_DIR) + "/" + name;
  if (!LittleFS.exists(path)) {
    server.send(404, "text/plain", "Recording not found.");
    return;
  }

  LittleFS.remove(path);
  server.sendHeader("Location", "/", true);
  server.send(303, "text/plain", "Deleted. Return to the recorder page.");
}

// ============================================================================
// WEB SERVER INITIALIZATION
// ============================================================================
void startWebServer() {
  WiFi.mode(WIFI_AP);

  if (!WiFi.softAP(AP_SSID, AP_PASSWORD)) {
    Serial.println("Wi-Fi access point start failed.");
    return;
  }

  // HTML Web Interface
  server.on("/", HTTP_GET, handleWebRoot);
  server.on("/download", HTTP_GET, handleWebDownload);
  server.on("/delete", HTTP_POST, handleWebDelete);

  // REST API Endpoints
  server.on("/api/status", HTTP_GET, handleApiStatus);
  server.on("/api/status", HTTP_OPTIONS, handleOptions);

  server.on("/api/recordings", HTTP_GET, handleApiRecordings);
  server.on("/api/recordings", HTTP_OPTIONS, handleOptions);

  server.on("/api/recordings/download", HTTP_GET, handleApiDownload);
  server.on("/api/recordings/download", HTTP_OPTIONS, handleOptions);

  server.on("/api/recordings/delete", HTTP_POST, handleApiDelete);
  server.on("/api/recordings/delete", HTTP_OPTIONS, handleOptions);
  server.on("/api/recordings", HTTP_DELETE, handleApiDelete);

  server.on("/api/recordings/rename", HTTP_POST, handleApiRename);
  server.on("/api/recordings/rename", HTTP_OPTIONS, handleOptions);

  server.on("/api/record/start", HTTP_POST, handleApiRecordStart);
  server.on("/api/record/start", HTTP_OPTIONS, handleOptions);

  server.on("/api/record/stop", HTTP_POST, handleApiRecordStop);
  server.on("/api/record/stop", HTTP_OPTIONS, handleOptions);

  server.onNotFound([]() {
    setCorsHeaders();
    if (server.method() == HTTP_OPTIONS) {
      server.send(204);
      return;
    }
    server.send(404, "application/json", "{\"ok\":false,\"error\":\"Endpoint not found\"}");
  });

  server.begin();

  Serial.println();
  Serial.println("Wi-Fi access point started.");
  Serial.print("Wi-Fi name: ");
  Serial.println(AP_SSID);
  Serial.print("Wi-Fi password: ");
  Serial.println(AP_PASSWORD);
  Serial.print("Web address: http://");
  Serial.println(WiFi.softAPIP());
}

// ============================================================================
// SERIAL COMMANDS (PRESERVED EXACTLY AS TESTED)
// ============================================================================
void listRecordings() {
  File dir = LittleFS.open(RECORD_DIR);
  if (!dir || !dir.isDirectory()) {
    Serial.println("Recordings directory is unavailable.");
    return;
  }

  Serial.println();
  Serial.println("----- SAVED RECORDINGS -----");

  File entry = dir.openNextFile();
  while (entry) {
    Serial.print(entry.name());
    Serial.print(" | ");
    Serial.print(entry.size());
    Serial.println(" bytes");
    entry.close();
    entry = dir.openNextFile();
  }

  Serial.println("----------------------------");
  dir.close();
}

void processCommand() {
  if (!Serial.available()) return;

  String command = Serial.readStringUntil('\n');
  command.trim();
  command.toUpperCase();

  if (command == "START") {
    startRecording();
  } else if (command == "STOP") {
    stopRecording("Recording stopped by user.");
  } else if (command == "STATUS") {
    if (isRecording) {
      uint32_t elapsed = (millis() - recordingStartMs) / 1000UL;
      uint32_t remaining = (elapsed < MAX_RECORD_SECONDS) ? MAX_RECORD_SECONDS - elapsed : 0;
      Serial.printf("RECORDING; %lu seconds remaining\n", (unsigned long)remaining);
    } else {
      Serial.println("READY");
    }
  } else if (command == "LIST") {
    listRecordings();
  } else {
    Serial.println("Serial commands: START, STOP, STATUS, LIST");
    Serial.println("REST API available on http://192.168.4.1/api/status");
  }
}

// ============================================================================
// ARDUINO SETUP & LOOP
// ============================================================================
void setup() {
  Serial.begin(115200);
  Serial.setTimeout(30);
  delay(500);

  Wire.begin(OLED_SDA, OLED_SCL);

  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDRESS)) {
    Serial.println("OLED initialization failed. Check address/wiring.");
    while (true) delay(100);
  }

  showMessage("VOICE RECORDER", "INITIALIZING");

  if (!LittleFS.begin(false)) {
    Serial.println("LittleFS initialization failed!");
    showMessage("STORAGE ERROR", "LittleFS failed");
    while (true) delay(100);
  }

  Serial.println("LittleFS mounted successfully!");

  if (!LittleFS.exists(RECORD_DIR) && !LittleFS.mkdir(RECORD_DIR)) {
    Serial.println("Could not create /recordings directory.");
    showMessage("FOLDER ERROR", "LittleFS");
    while (true) delay(100);
  }

  if (!setupI2S()) {
    Serial.println("I2S initialization failed. Check INMP441 wiring.");
    showMessage("I2S ERROR", "Check microphone");
    while (true) delay(100);
  }

  startWebServer();

  showMessage("VOICE RECORDER", "READY");

  Serial.println();
  Serial.println("==================================================");
  Serial.println("ESP32 INMP441 VOICE RECORDER READY");
  Serial.println("Storage: LittleFS internal flash");
  Serial.printf("Format: WAV, %d Hz, 16-bit, mono\n", SAMPLE_RATE);
  Serial.printf("Max recording: %d seconds\n", MAX_RECORD_SECONDS);
  Serial.println("Web page: http://192.168.4.1");
  Serial.println("REST API: http://192.168.4.1/api/status");
  Serial.println("==================================================");
}

void loop() {
  server.handleClient();
  processCommand();
  recordAudio();
}
