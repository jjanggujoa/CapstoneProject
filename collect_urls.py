import time
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# ==========================================
# ⚙️ 전략 수정: 기준 완화 & 전국 확장
# ==========================================
TARGET_MAX_RATING = 3.9  # 기준을 3.9로 올려서 리뷰 많은 호텔도 포함시킴
# 일본 주요 도시 + 부정 리뷰가 나오기 쉬운 키워드 조합
SEARCH_KEYWORDS = [
    "Tokyo Capsule Hotel", "Tokyo Cheap Hotel", "Tokyo Hostel",
    "Osaka Capsule Hotel", "Osaka Cheap Hotel", "Osaka Hostel",
    "Fukuoka Capsule Hotel", "Fukuoka Business Hotel",
    "Kyoto Guesthouse", "Kyoto Cheap Hotel",
    "Sapporo Hotel", "Okinawa Hotel",
    "Nagoya Business Hotel"
]
# ==========================================

options = Options()
options.add_argument("--lang=en")
driver = webdriver.Chrome(options=options)

collected_urls = set()

print(f"🎯 전략: 일본 전국 대상 / 평점 {TARGET_MAX_RATING}점 이하 / 가성비 숙소 위주 수집")

for keyword in SEARCH_KEYWORDS:
    # 검색어 URL 생성 (구글맵 검색 쿼리)
    search_url = f"https://www.google.com/maps/search/{keyword.replace(' ', '+')}"

    print(f"\n🔍 검색 중: '{keyword}' ...")
    driver.get(search_url)
    time.sleep(5)

    try:
        scrollable_div = driver.find_element(By.CSS_SELECTOR, "div[role='feed']")
    except:
        print("⚠️ 결과 없음 혹은 로딩 실패. 다음 키워드로.")
        continue

    # 키워드 하나당 스크롤 30번 (충분히 많이)
    for i in range(30):
        cards = driver.find_elements(By.CSS_SELECTOR, "div.Nv2PK")

        for card in cards:
            try:
                # URL 추출
                link_elem = card.find_element(By.TAG_NAME, "a")
                url = link_elem.get_attribute("href")
                if not url or "/maps/place/" not in url: continue
                clean_url = url.split("?")[0]

                if clean_url in collected_urls: continue

                # 평점 확인
                try:
                    score_text = card.find_element(By.CSS_SELECTOR, "span.MW4etd").text
                    score = float(score_text)
                except:
                    score = 0.0  # 평점 없으면 신규 호텔일 수 있으니 일단 수집

                # 4.2점 이하면 수집 (기준 완화)
                if score <= TARGET_MAX_RATING:
                    collected_urls.add(clean_url)
                    # 로그 줄이기 (너무 많이 뜨면 정신없음)
                    if len(collected_urls) % 10 == 0:
                        print(f"   Op.. 현재 총 {len(collected_urls)}개 URL 확보 중")
            except:
                pass

        # 스크롤
        driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scrollable_div)
        time.sleep(1.5)

    print(f"   ➡️ '{keyword}' 완료. 누적 URL: {len(collected_urls)}개")

# 저장
with open("data/raw/hotel_urls.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for u in collected_urls:
        writer.writerow([u])

driver.quit()
print(f"\n🔥 [최종 완료] 총 {len(collected_urls)}개의 호텔 URL을 저장했습니다.")
print("👉 이제 crawler_50k_final.py를 실행해서 리뷰를 긁어모으세요!")
