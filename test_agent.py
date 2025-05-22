def test_basic_crawling():
    from crawler import crawl_and_extract
    data = crawl_and_extract("https://help.zluri.com", max_pages=3)
    assert isinstance(data, dict) and len(data) > 0

def test_embedding_search():
    from embedder import Embedder
    sample_data = {"url1": "Zluri supports many integrations including Slack and Zoom."}
    emb = Embedder()
    emb.build_index(sample_data)
    result = emb.search("What integrations are available?")
    assert result and 'Slack' in result[0][0]
