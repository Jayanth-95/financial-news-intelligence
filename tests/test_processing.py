from utils.extract import clean_text
from utils.filtering import is_market_relevant
from utils.parse import extract_currency_values, extract_events, extract_percentages, extract_tickers

def run():
    assert clean_text("  Revenue\n\n rose \t 15%  ") == "Revenue rose 15%"
    assert is_market_relevant("The company reported strong earnings.")
    assert not is_market_relevant("A sunny day in Hyderabad.")
    text = "AAPL gained 12.5% after earnings. Revenue reached $2,500."
    assert extract_percentages(text) == ["12.5%"]
    assert extract_currency_values(text) == ["$2,500"]
    assert "AAPL" in extract_tickers(text)
    assert "earnings" in extract_events(text)

if __name__ == "__main__":
    run()
