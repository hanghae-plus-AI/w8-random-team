import requests
from bs4 import BeautifulSoup

# 크롤링할 웹 페이지 URL
url = "https://spartacodingclub.kr/blog"
url2 = "https://spartacodingclub.kr"

# HTTP 요청 및 응답 받기
response = requests.get(url)
response.raise_for_status()  # 응답 에러 확인


# 페이지 파싱
soup = BeautifulSoup(response.text, "html.parser")

# 큰 메뉴 검색 (예: 클래스가 'blog-menu'인 div로 가정)
section1 = soup.find("section", class_="css-p026og")
atags1 = section1.findAll("a")
for a in atags1:
    print(a.get("href"))

section2 = soup.find("section", class_="css-17r5dgq")
atags2 = section2.findAll("a")
for a in atags2:
    print(a.get("href"))
scrap_urls = [
    *[url2 + atag.get("href")[1:] for atag in atags1],
    *[url2 + atag.get("href") for atag in atags2],
]


from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings

contents = []
for index, url in enumerate(scrap_urls):
    print("--------", index)
    try:
        # HTTP 요청 및 응답 받기
        response = requests.get(url)
        response.raise_for_status()  # 응답 에러 확인
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        response = None  # 오류가 발생했을 때 None으로 설정

    # 페이지 파싱 (response가 성공적으로 받아졌을 때만)

    if response:
        soup = BeautifulSoup(response.text, "html.parser")
        if index == 4:
            # print(response.text)
            content = soup.findAll("section")
            print([section.get("class") for section in content])
        if soup.find("section", class_="css-18vt64m") is not None:
            content = soup.find("section", class_="css-18vt64m").get_text(strip=True)
            contents.append(content)

documents = [Document(page_content=chunk) for chunk in contents]

print(documents)
from dotenv import load_dotenv

load_dotenv()
vectorstore = FAISS.from_documents(documents=documents, embedding=OpenAIEmbeddings())
vectorstore.save_local("vector_stores/randomteam_blog")
