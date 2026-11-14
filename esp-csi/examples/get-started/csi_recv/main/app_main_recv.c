#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdint.h>
#include "esp_event.h"
#include "mqtt_client.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"

#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_now.h"

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
#define WIFI_BW_HT20 WIFI_BW20
#define WIFI_BW_HT40 WIFI_BW40
#endif

#define CONFIG_LESS_INTERFERENCE_CHANNEL   11
#define WIFI_STA_SSID                      "Minh Duc MSI"
#define WIFI_STA_PASSWORD                  "Minh Duc MSI"
#define MQTT_BROKER_IP                     "192.168.137.1"
#define MQTT_TOPIC_CSI                     "esp32/csi_data"
#define CSI_QUEUE_LENGTH                   128
#define CSI_QUEUE_MAX_LEN                  512
#define CSI_MQTT_TASK_STACK_SIZE           6144

#define CONFIG_WIFI_BANDWIDTH              WIFI_BW_HT20
#define CONFIG_ESP_NOW_PHYMODE             WIFI_PHY_MODE_HT20
#define CONFIG_ESP_NOW_RATE                WIFI_PHY_RATE_MCS0_LGI

// Địa chỉ MAC của board phát (TX) dùng để lọc dữ liệu đầu vào
static const uint8_t CONFIG_CSI_SEND_MAC_PREFIX[5] = {0x1a, 0x00, 0x00, 0x00, 0x00};


static const char *TAG = "csi_recv";
static esp_mqtt_client_handle_t s_mqtt_client = NULL;
static volatile bool s_mqtt_connected = false;
static QueueHandle_t s_csi_queue = NULL;
static volatile uint32_t s_csi_dropped_packets = 0;

typedef struct {
    uint8_t tx_id;
    uint32_t seq;
    int8_t rssi;
    uint16_t len;
    uint8_t csi[CSI_QUEUE_MAX_LEN];
} csi_mqtt_packet_t;

static void mqtt_event_handler(void *handler_args, esp_event_base_t base,
                               int32_t event_id, void *event_data)
{
    (void)handler_args;
    (void)base;
    (void)event_data;

    if (event_id == MQTT_EVENT_CONNECTED) {
        s_mqtt_connected = true;
        ESP_LOGI(TAG, "Connected to MQTT broker");
    } else if (event_id == MQTT_EVENT_DISCONNECTED) {
        s_mqtt_connected = false;
        ESP_LOGW(TAG, "Disconnected from MQTT broker");
    } else if (event_id == MQTT_EVENT_ERROR) {
        ESP_LOGE(TAG, "MQTT connection error");
    }
}

static void mqtt_init(void)
{
    static char broker_uri[64];

    int uri_length = snprintf(broker_uri, sizeof(broker_uri),
                              "mqtt://%s:1883", MQTT_BROKER_IP);
    if (uri_length < 0 || (size_t)uri_length >= sizeof(broker_uri)) {
        ESP_LOGE(TAG, "MQTT broker URI is too long");
        abort();
    }

    const esp_mqtt_client_config_t mqtt_config = {
        .broker.address.uri = broker_uri,
        .network.disable_auto_reconnect = false,
        .network.timeout_ms = 30000,
        .buffer.out_size = 4096,
    };
    s_mqtt_client = esp_mqtt_client_init(&mqtt_config);
    if (s_mqtt_client == NULL) {
        ESP_LOGE(TAG, "Failed to initialize MQTT client");
        abort();
    }

    ESP_ERROR_CHECK(esp_mqtt_client_register_event(
                        s_mqtt_client, ESP_EVENT_ANY_ID, mqtt_event_handler, NULL));
    ESP_ERROR_CHECK(esp_mqtt_client_start(s_mqtt_client));
}

static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data)
{
    (void)arg;
    (void)event_data;

    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "Wi-Fi disconnected; reconnecting");
        ESP_ERROR_CHECK(esp_wifi_connect());
    }
}

static void ip_event_handler(void *arg, esp_event_base_t event_base,
                             int32_t event_id, void *event_data)
{
    (void)arg;

    if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        const ip_event_got_ip_t *event = event_data;
        ESP_LOGI(TAG, "Wi-Fi connected, IP: " IPSTR, IP2STR(&event->ip_info.ip));

        // Chỉ khởi tạo MQTT sau khi đã nhận IP từ Mobile Hotspot
        if (s_mqtt_client == NULL) {
            mqtt_init();
        }
    }
}

static void wifi_sta_connect(void)
{
    wifi_config_t wifi_config = {
        .sta = {
            .ssid = WIFI_STA_SSID,
            .password = WIFI_STA_PASSWORD,
            .channel = CONFIG_LESS_INTERFERENCE_CHANNEL,
            .scan_method = WIFI_FAST_SCAN,
        },
    };

    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_connect());
}

static void csi_mqtt_task(void *arg)
{
    (void)arg;

    static char json[160 + (CSI_QUEUE_MAX_LEN / 2) * 5];
    csi_mqtt_packet_t packet;
    uint32_t reported_drops = 0;

    while (true) {
        if (xQueueReceive(s_csi_queue, &packet, portMAX_DELAY) != pdTRUE) {
            continue;
        }

        const uint32_t dropped_packets = s_csi_dropped_packets;
        if (dropped_packets != reported_drops) {
            ESP_LOGW(TAG, "Dropped %lu CSI packets (queue full or CSI too long)",
                     (unsigned long)(dropped_packets - reported_drops));
            reported_drops = dropped_packets;
        }

        if (!s_mqtt_connected) {
            continue;
        }

        const size_t csi_count = packet.len / 2;
        int written = snprintf(json, sizeof(json),
                       "{\"tx_id\":%u,\"seq\":%lu,\"rssi\":%d,\"len\":%u,\"csi\":[",
                       packet.tx_id, (unsigned long)packet.seq,
                       packet.rssi, (unsigned)packet.len);
        if (written < 0 || (size_t)written >= sizeof(json)) {
            continue;
        }

        size_t json_length = (size_t)written;
        bool format_error = false;
        for (size_t i = 0; i < csi_count; i++) {
            const int8_t i_component = (int8_t)packet.csi[i * 2];
            const int8_t q_component = (int8_t)packet.csi[i * 2 + 1];
            const int amplitude = (int)(sqrtf((float)(i_component * i_component +
                                                        q_component * q_component)) + 0.5f);

            written = snprintf(json + json_length, sizeof(json) - json_length,
                               "%s%d", i == 0 ? "" : ",", amplitude);
            if (written < 0 || (size_t)written >= sizeof(json) - json_length) {
                format_error = true;
                break;
            }
            json_length += (size_t)written;
        }
        if (format_error) {
            continue;
        }

        written = snprintf(json + json_length, sizeof(json) - json_length, "]}");
        if (written < 0 || (size_t)written >= sizeof(json) - json_length) {
            continue;
        }
        json_length += (size_t)written;

        // Đẩy bất đồng bộ vào hàng đợi xuất MQTT để tránh treo luồng khi Wi-Fi trễ
        int message_id = esp_mqtt_client_publish(s_mqtt_client, MQTT_TOPIC_CSI,
                                         json, (int)json_length, 0, 0);
        if (message_id < 0) {
            ESP_LOGW(TAG, "Failed to publish CSI packet %lu",
                    (unsigned long)packet.seq);
        }
    }
}

static void csi_mqtt_queue_init(void)
{
    s_csi_queue = xQueueCreate(CSI_QUEUE_LENGTH, sizeof(csi_mqtt_packet_t));
    ESP_ERROR_CHECK(s_csi_queue != NULL ? ESP_OK : ESP_ERR_NO_MEM);

    BaseType_t task_created = xTaskCreate(csi_mqtt_task, "csi_mqtt",
                                          CSI_MQTT_TASK_STACK_SIZE, NULL, 5, NULL);
    ESP_ERROR_CHECK(task_created == pdPASS ? ESP_OK : ESP_ERR_NO_MEM);
}

static void wifi_init(void)
{
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_netif_create_default_wifi_sta() != NULL ? ESP_OK : ESP_FAIL);
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                               wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                               ip_event_handler, NULL));

    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_STA, CONFIG_WIFI_BANDWIDTH));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_NONE));
    // Đã xóa lệnh esp_wifi_set_mac ở đây để RX giữ nguyên MAC gốc, tránh trùng với TX
}

static void wifi_esp_now_init(esp_now_peer_info_t peer)
{
    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_set_pmk((uint8_t *)"pmk1234567890123"));
    esp_now_rate_config_t rate_config = {
        .phymode = CONFIG_ESP_NOW_PHYMODE,
        .rate = CONFIG_ESP_NOW_RATE,
        .ersu = false,
        .dcm = false
    };
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));
    ESP_ERROR_CHECK(esp_now_set_peer_rate_config(peer.peer_addr, &rate_config));
}

static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    (void)ctx;

    if (!info || !info->buf) {
        return;
    }

    // Lọc đúng MAC của TX đầu tiên
    if (memcmp(info->mac, CONFIG_CSI_SEND_MAC_PREFIX, 5) != 0 ||
        info->mac[5] < 1 || info->mac[5] > 3) {
        return;
    }

    if (info->len == 0 || !info->payload || info->payload_len < 15 + sizeof(uint32_t)) {
        return;
    }

    if (s_csi_queue == NULL) {
        return;
    }

    uint32_t rx_id = 0;
    // Dịch 15 byte header để đọc đúng biến count của TX
    memcpy(&rx_id, info->payload + 15, sizeof(rx_id));

    csi_mqtt_packet_t packet = {
        .tx_id = info->mac[5],
        .seq = rx_id,
        .rssi = info->rx_ctrl.rssi,
        .len = (uint16_t)info->len,
    };

    if (info->len > sizeof(packet.csi)) {
        s_csi_dropped_packets++;
        return;
    }

    memcpy(packet.csi, info->buf, info->len);

    csi_mqtt_packet_t dummy_packet;
    if (uxQueueSpacesAvailable(s_csi_queue) == 0) {
        // Lấy ra gói cũ nhất nhưng không quan tâm nội dung, để giải phóng 1 ô nhớ
        xQueueReceive(s_csi_queue, &dummy_packet, 0);
        s_csi_dropped_packets++; // Đếm số gói cũ bị loại bỏ
    }

    // Đưa gói mới nhất vào hàng đợi
    if (xQueueSend(s_csi_queue, &packet, 0) != pdTRUE) {
        s_csi_dropped_packets++;
    }
}

static void wifi_csi_init(void)
{
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));

    wifi_csi_config_t csi_config = {
        .lltf_en           = true,
        .htltf_en          = true,
        .stbc_htltf2_en    = true,
        .ltf_merge_en      = true,
        .channel_filter_en = true,
        .manu_scale        = false,
        .shift             = false,
    };
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    wifi_init();
    wifi_sta_connect();

    esp_now_peer_info_t peer = {
        .channel   = CONFIG_LESS_INTERFERENCE_CHANNEL,
        .ifidx     = WIFI_IF_STA,
        .encrypt   = false,
        .peer_addr = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
    };

    wifi_esp_now_init(peer);
    csi_mqtt_queue_init();
    wifi_csi_init();
}