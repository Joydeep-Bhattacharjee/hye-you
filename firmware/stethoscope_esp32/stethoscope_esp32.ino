// Digital stethoscope - INMP441 I2S mic -> 16 kHz int16 audio over USB serial.
// ESP32 core 2.0.x (legacy I2S API). Board: "ESP32 Dev Module".
#include <driver/i2s.h>

#define I2S_SCK   26
#define I2S_WS    25
#define I2S_SD    33
#define I2S_PORT  I2S_NUM_0
#define SAMPLE_RATE  16000
#define BUF_SAMPLES  256

const uint8_t SYNC[4] = {0xA5, 0x5A, 0xA5, 0x5A};
int32_t raw[BUF_SAMPLES];
int16_t out[BUF_SAMPLES];

void i2sInit() {
  i2s_config_t cfg = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = 8,
    .dma_buf_len = 256,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };
  i2s_driver_install(I2S_PORT, &cfg, 0, NULL);
  i2s_pin_config_t pins = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };
  i2s_set_pin(I2S_PORT, &pins);
  i2s_zero_dma_buffer(I2S_PORT);
}

void setup() {
  Serial.begin(921600);
  delay(300);
  Serial.println("ESP32_STETHO_READY");
  i2sInit();
}

void loop() {
  size_t bytesRead = 0;
  i2s_read(I2S_PORT, raw, sizeof(raw), &bytesRead, portMAX_DELAY);
  int n = bytesRead / 4;
  for (int i = 0; i < n; i++) {
    int32_t v = raw[i] >> 11;
    if (v > 32767) v = 32767;
    if (v < -32768) v = -32768;
    out[i] = (int16_t)v;
  }
  Serial.write(SYNC, 4);
  uint16_t cnt = (uint16_t)n;
  Serial.write((uint8_t*)&cnt, 2);
  Serial.write((uint8_t*)out, n * 2);
}
