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

# 點擊 "資料圖表展示" 按鈕
data_button = driver.find_element(By.XPATH, "//button[contains(text(), '資料圖表展示')]")
data_button.click()

# XPath 篩選條件：匹配 section.zh-TW 下未隱藏的 datetime-tool-content
datetime_content = driver.find_element(By.XPATH, "//section[@class='zh-TW']//div[@class='lightbox-tool-type-container' and not(contains(@style, 'display: none'))]//div[@class='datetime-tool-content']")

# 定位目標 svg 圖標
date_icon = datetime_content.find_element(By.CLASS_NAME, "datetime-tool-icon")

# 模擬滑鼠單擊行為
action = ActionChains(driver)
action.move_to_element(date_icon).pause(0.5).click().perform()

# 設置日期（需要確保格式符合）
date_year = datetime_content.find_element(By.CLASS_NAME, "vdatetime-popup__year")
date_year.click()  # 可加進一步操作設置具體年份

# 定位年份選擇器
year_picker = datetime_content.find_element(By.CLASS_NAME, "vdatetime-year-picker")

# 獲取所有年份項目
year_items = year_picker.find_elements(By.CLASS_NAME, "vdatetime-year-picker__item")

# 點擊目標年份
for year_item in year_items:
    if year_item.text.strip() == "2024":
        print("找到年份 2024，點擊...")
        # 點擊目標年份，模擬滑鼠單擊行為
        action = ActionChains(driver)
        action.move_to_element(year_item).pause(0.5).click().perform()
        break

# 定位月份選擇器
date_date = datetime_content.find_element(By.CLASS_NAME, "vdatetime-popup__date")
date_date.click()  # 可加進一步操作設置具體月份

month_picker = datetime_content.find_element(By.CLASS_NAME, "vdatetime-month-picker")

# 獲取所有月份項目
month_items = month_picker.find_elements(By.CLASS_NAME, "vdatetime-month-picker__item")

# 點擊目標月份
for month_item in month_items:
    if month_item.text.strip() == "1月":
        print("找到月份 1月，點擊...")
        # 點擊目標月份，模擬滑鼠單擊行為
        action = ActionChains(driver)
        action.move_to_element(month_item).pause(0.5).click().perform()
        break

############################## OK ##############################
# 點擊下載 CSV 按鈕
# time.sleep(2)
# csv_button = driver.find_element(By.XPATH, "//div[@class='lightbox-tool-type-ctrl-btn' and contains(., 'CSV下載')]")
# csv_button.click()

# # # 等待下載完成後關閉
time.sleep(5)
############################## OK ##############################
driver.quit()


# print("日期選擇器是否顯示:", date_picker.is_displayed())
# print("日期選擇器是否可互動:", date_picker.is_enabled())
