import time
import csv
import re
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# ==========================================
# ⚙️ 설정값
# ==========================================
TOTAL_TARGET = 50000
PER_HOTEL_LIMIT = 300
SAVE_FILE_NAME = "data/raw/japan_hotel_reviews_50k.csv"
# ⭐ 방금 수집하신 URL 파일명으로 정확히 지정
URL_FILE_NAME = "data/raw/hotel_urls.csv"


# ==========================================

# 1. 리뷰 버튼 클릭
def click_review_button(driver):
    review_keywords = ["리뷰", "Review", "Reviews", "クチコミ", "口コミ"]
    for key in review_keywords:
        try:
            xpath = f"//button[contains(@aria-label, '{key}')] | //div[contains(text(), '{key}')] | //button[contains(., '{key}')]"
            btn = driver.find_element(By.XPATH, xpath)
            driver.execute_script("arguments[0].click();", btn)
            time.sleep(3)
            return True
        except:
            pass
    js_selectors = ["button[jsaction='pane.reviewChart.moreReviews']", "button[jsaction*='moreReviews']",
                    "button[jsaction*='review']"]
    for sel in js_selectors:
        try:
            btn = driver.find_element(By.CSS_SELECTOR, sel)
            driver.execute_script("arguments[0].click();", btn)
            time.sleep(3)
            return True
        except:
            pass
    return False


# 2. 스크롤 박스 찾기
def find_scroll_box(driver):
    try:
        candidates = driver.find_elements(By.CSS_SELECTOR, "div.m6QErb")
        best_box = None
        max_reviews = 0
        for box in candidates:
            reviews = box.find_elements(By.CSS_SELECTOR, "div.jftiEf")
            review_count = len(reviews)
            overflow_y = driver.execute_script("return window.getComputedStyle(arguments[0]).overflowY;", box)
            scroll_height = driver.execute_script("return arguments[0].scrollHeight;", box)
            client_height = driver.execute_script("return arguments[0].clientHeight;", box)
            if review_count > 0 and (overflow_y in ['scroll', 'auto'] or scroll_height > client_height):
                if review_count >= max_reviews:
                    max_reviews = review_count
                    best_box = box
        return best_box if best_box else driver.find_element(By.TAG_NAME, "body")
    except:
        return None


# 3. 스크롤 로딩
def scroll_reviews(driver, scroll_box, limit=500):
    last_count = len(driver.find_elements(By.CSS_SELECTOR, "div.jftiEf"))
    retry_count = 0
    print(f"   🔄 로딩 중...", end="", flush=True)
    while True:
        if last_count >= limit:
            print(" (목표 달성)")
            break
        driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight;", scroll_box)
        time.sleep(1.5)
        new_count = len(driver.find_elements(By.CSS_SELECTOR, "div.jftiEf"))
        if new_count > last_count:
            last_count = new_count
            retry_count = 0
            if new_count % 50 == 0: print(".", end="", flush=True)
        else:
            retry_count += 1
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight - 300;", scroll_box)
            time.sleep(0.5)
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight;", scroll_box)
            if retry_count >= 5:
                print(" (더 이상 없음)")
                break
    print()


# 4. ⭐ [핵심 수정] HTML 소스코드 원문 검색 방식
# 요소를 찾는 게 아니라, HTML 텍스트를 통째로 가져와서 "별표 X개"라는 글자를 찾습니다.
def extract_reviews(driver):
    elems = driver.find_elements(By.CSS_SELECTOR, "div.jftiEf")
    results = []

    for e in elems:
        try:
            # 더보기
            try:
                more_btn = e.find_element(By.CSS_SELECTOR, "button[jsaction*='expand']")
                driver.execute_script("arguments[0].click();", more_btn)
                time.sleep(0.05)
            except:
                pass

            # 텍스트
            text = ""
            try:
                text = e.find_element(By.CSS_SELECTOR, "span.wiI7pd").text.strip()
            except:
                pass

            # ⭐ 별점 추출 (HTML 소스 텍스트 검색)
            rating = None
            try:
                # 해당 리뷰 덩어리의 HTML 소스를 문자열로 가져옴
                html_source = e.get_attribute('outerHTML')

                # 정규표현식으로 "별표 1개" ~ "별표 5개" 패턴을 직접 찾음
                # aria-label="별표 5개" << 이 패턴을 찾습니다.
                match = re.search(r'별표\s*(\d)\s*개', html_source)

                if match:
                    rating = int(match.group(1))
                else:
                    # 혹시 모르니 "5 stars" 같은 영어 패턴도 대비
                    match_en = re.search(r'(\d)\s*stars', html_source)
                    if match_en:
                        rating = int(match_en.group(1))
                    else:
                        # "평점: 5/5" 패턴 대비
                        match_score = re.search(r'aria-label=".*?(\d)\s*/\s*5.*?"', html_source)
                        if match_score:
                            rating = int(match_score.group(1))

            except:
                pass

            if text:
                results.append({"review": text, "rating": rating})
        except:
            pass
    return results


# 메인 실행
if __name__ == "__main__":
    options = Options()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")
    # ⭐ 한국어 설정 (필수)
    options.add_argument("--lang=ko")

    driver = webdriver.Chrome(options=options)

    urls = []
    if os.path.exists(URL_FILE_NAME):
        with open(URL_FILE_NAME, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]
    else:
        print(f"❌ {URL_FILE_NAME} 파일이 없습니다!")
        exit()

    # 파일 초기화 (새로 시작)
    with open(SAVE_FILE_NAME, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["review", "rating"])
        writer.writeheader()

    current_count = 0
    print(f"🎯 전체 목표: {TOTAL_TARGET}개")

    for idx, url in enumerate(urls):
        if current_count >= TOTAL_TARGET:
            print(f"\n🎉 목표 달성 완료!")
            break

        print(f"\n[{idx + 1}/{len(urls)}] 호텔 접속 중...")
        try:
            driver.get(url)
            time.sleep(4)
            if not click_review_button(driver): continue

            scroll_box = find_scroll_box(driver)
            if scroll_box:
                scroll_reviews(driver, scroll_box, limit=PER_HOTEL_LIMIT)
                reviews = extract_reviews(driver)

                if reviews:
                    with open(SAVE_FILE_NAME, "a", newline="", encoding="utf-8-sig") as f:
                        writer = csv.DictWriter(f, fieldnames=["review", "rating"])
                        for r in reviews:
                            writer.writerow(r)

                    added = len(reviews)
                    current_count += added
                    print(f"✅ 저장 완료: +{added}개 (총 {current_count}개)")

                    # ✅ 확인용 로그 (제발 나와라)
                    if added > 0:
                        first_rating = reviews[0].get('rating')
                        print(f"   👀 [최종확인] 별점: {first_rating} / 리뷰: {reviews[0]['review'][:10]}...")
            else:
                print("❌ 스크롤 박스 없음")
        except Exception as e:
            print(f"⚠️ 에러: {e}")
            continue

    driver.quit()
    print(f"\n🔥 완료! 저장 경로: {SAVE_FILE_NAME}")