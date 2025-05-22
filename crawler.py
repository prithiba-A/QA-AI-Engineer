import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def extract_links(base_url, soup):
    links = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag['href']
        full_url = urljoin(base_url, href)
        if base_url in full_url:
            links.add(full_url)
    return links

def extract_text(soup):
    for tag in soup(['script', 'style', 'header', 'footer', 'nav']):
        tag.decompose()
    return soup.get_text(separator=' ', strip=True)

def crawl_and_extract(start_url, max_pages=50):
    visited = set()
    to_visit = [start_url]
    data = {}

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited or not is_valid_url(url):
            continue

        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            text = extract_text(soup)
            data[url] = text
            visited.add(url)
            to_visit.extend(extract_links(start_url, soup) - visited)
        except Exception as e:
            print(f"Error processing {url}: {e}")

    return data
