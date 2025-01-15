from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import time

# 使用 Service 類別
driver = webdriver.Chrome()

# 開啟目標網站
driver.get("https://codis.cwa.gov.tw/StationData")
driver.maximize_window()

# 等待頁面載入
WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CLASS_NAME, "form-control"))
)

# 找到輸入框，輸入站名和站號 "臺南 467410"
station_input = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, "input.form-control"))
)
station_input.click()
station_input.clear()
station_input.send_keys("臺南 (467410)")  # 輸入站名站號

# 等待紅色地圖圖標出現
time.sleep(2)
map_marker = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CLASS_NAME, "leaflet-marker-icon"))
)

# 模擬滑鼠單擊行為
action = ActionChains(driver)
action.move_to_element(map_marker).pause(0.5).click().perform()

# 等待彈出框顯示
popup = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CLASS_NAME, "leaflet-popup-content-wrapper"))
)

print("彈出框已成功顯示！")

# 點擊 "資料圖表展示" 按鈕
data_button = driver.find_element(By.XPATH, "//button[contains(text(), '資料圖表展示')]")
data_button.click()

# # 選擇日期範圍
# time.sleep(2)
# date_picker = driver.find_element(By.CLASS_NAME, "vdatetime-input")
# date_picker.click()

# # 設置日期（需要確保格式符合）
# time.sleep(1)
# date_year = driver.find_element(By.CLASS_NAME, "vdatetime-popup__year")
# date_year.click()  # 可加進一步操作設置具體年份
# date_confirm = driver.find_element(By.CLASS_NAME, "vdatetime-popup__actions")
# date_confirm.click()

# # 點擊下載 CSV 按鈕
# time.sleep(2)
# csv_button = driver.find_element(By.XPATH, "//a[contains(@href, '.csv')]")
# csv_button.click()

# # 等待下載完成後關閉
# time.sleep(5)
driver.quit()
