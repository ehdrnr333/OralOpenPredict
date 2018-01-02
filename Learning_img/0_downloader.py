from lxml.html import parse
from io import StringIO
import requests
import urllib.request as req

def search_img(keyword):
    url = 'https://www.google.co.kr/search?q='+keyword+'&source=lnms&tbm=isch&sa=X&ved=0ahUKEwic-taB9IXVAhWDHpQKHXOjC14Q_AUIBigB&biw=1842&bih=990'
    text = requests.get(url, headers={'user-agent': ':Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}).text
    text_source = StringIO(text)
    parsed = parse(text_source)
    doc = parsed.getroot()
    imgs = doc.findall('.//img')

    img_list = []   # 이미지 경로가 담길 list
    for a in imgs:
        img_list.append(a.get('data-src'))
    return img_list

def download_list(img_list, dir):
    i = 1
    save_dir = "./eval/" + dir
    for img_url in img_list:
        path = save_dir + "/" + str(i) + "_" + dir + ".jpg"
        if img_url is not None:
            req.urlretrieve(img_url, path)
            print(">>image:"+dir+" "+str(i)+" download")
            i = i+1
        if i>100:
            break

def keyword_scrap(keyword, dir):
    img_list = search_img(keyword)
    download_list(img_list, dir)
    print(":::image:"+keyword+"finish")

if __name__ == '__main__':
    keyword_scrap("입 벌린 짤", "open_temp")
    keyword_scrap("입 다문 짤", "close_temp")