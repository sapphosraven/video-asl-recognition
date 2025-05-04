import os
import zipfile
import requests
import tqdm
from bs4 import BeautifulSoup

def download_opensubs(language='en'):
    """Download the OpenSubtitles ZIP for a specific language"""
    url = f"http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/raw/{language}.zip"
    fname = f"{language}.zip"
    print(f"Downloading: {url}")
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm.tqdm(total=total, unit='B', unit_scale=True, desc=fname) as bar:
        for data in resp.iter_content(1024):
            file.write(data)
            bar.update(len(data))
    return fname

def extract_sentences_from_zip(zip_path, max_files=200):
    """Extract <s> sentences from the first max_files XMLs in the zip"""
    all_sentences = []

    with zipfile.ZipFile(zip_path, 'r') as archive:
        xml_files = [f for f in archive.namelist() if f.endswith('.xml')]
        for file in tqdm.tqdm(xml_files[:max_files], desc="Parsing XML files"):
            with archive.open(file) as xml_file:
                try:
                    soup = BeautifulSoup(xml_file, 'html.parser')
                    sentences = [s.get_text().strip() for s in soup.find_all("s")]
                    all_sentences.extend(sentences)
                except Exception as e:
                    print(f"Error parsing {file}: {e}")
    return all_sentences

def save_sentences_to_txt(sentences, output_path="english_sentences.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        for s in sentences:
            if len(s.split()) >= 3:
                f.write(s.strip() + "\n")
    print(f"Saved {len(sentences)} sentences to {output_path}")

if __name__ == "__main__":
    language = 'en'
    zip_file = download_opensubs(language)
    sentences = extract_sentences_from_zip(zip_file, max_files=200)
    save_sentences_to_txt(sentences)
