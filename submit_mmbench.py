#!/usr/bin/env python3
"""Auto-submit MMBench xlsx files via headless Chromium."""
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def submit_file(driver, xlsx_path, name=""):
    """Submit a single xlsx file to MMBench and wait for result."""
    print(f"\n{'='*50}")
    print(f"Submitting: {name} ({os.path.basename(xlsx_path)})")
    print(f"{'='*50}")
    
    driver.get('https://mmbench.opencompass.org.cn/mmbench-submission')
    time.sleep(3)
    
    # Take screenshot for debugging
    driver.save_screenshot(f'/tmp/mmbench_{name}_1_loaded.png')
    
    # Find file upload input
    try:
        file_input = driver.find_element(By.CSS_SELECTOR, 'input[type="file"]')
        file_input.send_keys(xlsx_path)
        print(f"  File uploaded: {xlsx_path}")
        time.sleep(2)
        driver.save_screenshot(f'/tmp/mmbench_{name}_2_uploaded.png')
    except Exception as e:
        print(f"  ERROR finding file input: {e}")
        # Try alternative selectors
        inputs = driver.find_elements(By.TAG_NAME, 'input')
        print(f"  Found {len(inputs)} input elements")
        for i, inp in enumerate(inputs):
            print(f"    input[{i}]: type={inp.get_attribute('type')}, class={inp.get_attribute('class')}")
        return None
    
    # Find and click submit button
    try:
        buttons = driver.find_elements(By.TAG_NAME, 'button')
        submit_btn = None
        for btn in buttons:
            text = btn.text.lower()
            if 'submit' in text or '提交' in text or 'evaluate' in text:
                submit_btn = btn
                break
        
        if submit_btn:
            submit_btn.click()
            print(f"  Submit button clicked")
        else:
            print(f"  No submit button found. Buttons: {[b.text for b in buttons]}")
            return None
    except Exception as e:
        print(f"  ERROR clicking submit: {e}")
        return None
    
    # Wait for result
    print(f"  Waiting for evaluation result...")
    time.sleep(15)
    driver.save_screenshot(f'/tmp/mmbench_{name}_3_result.png')
    
    # Try to extract score from page
    page_text = driver.find_element(By.TAG_NAME, 'body').text
    print(f"  Page text (last 500 chars): {page_text[-500:]}")
    
    return page_text


def main():
    opts = Options()
    opts.add_argument('--headless')
    opts.add_argument('--no-sandbox')
    opts.add_argument('--disable-dev-shm-usage')
    opts.add_argument('--window-size=1920,1080')
    opts.binary_location = '/usr/bin/chromium'
    
    driver = webdriver.Chrome(options=opts)
    driver.implicitly_wait(10)
    
    # First, let's see what the page looks like
    driver.get('https://mmbench.opencompass.org.cn/mmbench-submission')
    time.sleep(5)
    driver.save_screenshot('/tmp/mmbench_initial.png')
    page_text = driver.find_element(By.TAG_NAME, 'body').text
    print("Initial page text:")
    print(page_text[:2000])
    print("\n--- HTML snippet ---")
    # Look for forms, inputs, buttons
    html = driver.page_source
    import re
    for tag in re.findall(r'<(input|button|form)[^>]*>', html, re.I):
        print(f"  {tag[:200]}")
    
    driver.quit()


if __name__ == "__main__":
    main()
