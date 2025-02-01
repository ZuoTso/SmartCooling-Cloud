from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import time
from datetime import datetime, timedelta

driver = webdriver.Chrome()

# Open the target website
driver.get("https://codis.cwa.gov.tw/StationData")
driver.maximize_window()

# Waiting for the page to load
WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CLASS_NAME, "form-control"))
)

# Find the input box and enter the station name and station number
station_input = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, "input.form-control"))
)
station_input.click()
station_input.clear()
station_input.send_keys("臺南 (467410)")  # Enter the station name and number

# Wait for the map icon to appear
time.sleep(2)
map_marker = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CLASS_NAME, "leaflet-marker-icon"))
)

# Simulate mouse click behavior
action = ActionChains(driver)
action.move_to_element(map_marker).pause(0.5).click().perform()

# Wait for the pop-up box to appear
popup = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CLASS_NAME, "leaflet-popup-content-wrapper"))
)

# Click "資料圖表展示" button
data_button = driver.find_element(By.XPATH, "//button[contains(text(), '資料圖表展示')]")
data_button.click()

# Match datetime-tool-content that is not hidden under section.zh-TW
datetime_content = driver.find_element(By.XPATH, "//section[@class='zh-TW']//div[@class='lightbox-tool-type-container' and not(contains(@style, 'display: none'))]//div[@class='datetime-tool-content']")

# Positioning target svg icon
date_icon = datetime_content.find_element(By.CLASS_NAME, "datetime-tool-icon")

# Simulate mouse click behavior
action = ActionChains(driver)
action.move_to_element(date_icon).pause(0.5).click().perform()

# Set initial date
current_date = datetime(2018, 12, 5)
target_year = str(current_date.year)
target_month = str(current_date.month)
target_day = str(current_date.day)

date_year = datetime_content.find_element(By.CLASS_NAME, "vdatetime-popup__year")
date_year.click()

# Positioning the year selector
year_picker = datetime_content.find_element(By.CLASS_NAME, "vdatetime-year-picker")

# Get all year items
year_items = year_picker.find_elements(By.CLASS_NAME, "vdatetime-year-picker__item")

# Click on the target year
for year_item in year_items:
    if year_item.text.strip() == target_year:
        print(f"找到年份 {target_year}，點擊...")
        action = ActionChains(driver)
        action.move_to_element(year_item).pause(0.5).click().perform()
        break

# Positioning the month picker
date_date = datetime_content.find_element(By.CLASS_NAME, "vdatetime-popup__date")
date_date.click()

month_picker = datetime_content.find_element(By.CLASS_NAME, "vdatetime-month-picker")

# Get all month items
month_items = month_picker.find_elements(By.CLASS_NAME, "vdatetime-month-picker__item")

# Click on the target month
for month_item in month_items:
    if month_item.text.strip() == f"{target_month}月":
        print(f"找到月份 {target_month}月，點擊...")
        action = ActionChains(driver)
        action.move_to_element(month_item).pause(0.5).click().perform()
        break

# Get all date items
date_items = datetime_content.find_elements(By.CLASS_NAME, "vdatetime-calendar__month__day")

# Click on the target date
for date_item in date_items:
    if date_item.text.strip() == target_day:
        print(f"找到日期 {target_day}，點擊...")
        action = ActionChains(driver)
        action.move_to_element(date_item).pause(0.5).click().perform()
        break

# Set end date
end_date = datetime(2018, 9, 25)

# Repeat the process to download data for each day until the end date.
while current_date >= end_date:
    # Wait for the download to complete
    time.sleep(3)
    print(f"日期: {current_date.strftime('%Y-%m-%d')}, CSV 下載")
    
    # Click the Download CSV button
    csv_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//div[@class='lightbox-tool-type-ctrl-btn' and contains(., 'CSV下載')]"))
    )
    csv_button.click()
    time.sleep(0.5)

    # Click the "Previous Page" button
    datetime_prev = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//section[@class='zh-TW']//div[@class='lightbox-tool-type-container' and not(contains(@style, 'display: none'))]//div[@class='datetime-tool-prev-next']"))
    )
    ActionChains(driver).move_to_element(datetime_prev).pause(0.5).click().perform()

    # Updated Date
    current_date -= timedelta(days=1)

driver.quit()
