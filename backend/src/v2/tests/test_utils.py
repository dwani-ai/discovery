from utils.text import clean_text

def test_clean_text():
    dirty = "Hello\u0000World\n\tControl\u007F"
    cleaned = clean_text(dirty)
    assert cleaned == "HelloWorld\n\tControl"