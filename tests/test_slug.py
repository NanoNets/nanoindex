from nanoindex.utils.slug import slugify


def test_basic():
    assert slugify("Hello World") == "hello-world"


def test_special_chars():
    assert slugify("3M Company (NYSE: MMM)") == "3m-company-nyse-mmm"


def test_underscores():
    assert slugify("some_file_name") == "some-file-name"


def test_truncation():
    long = "a" * 100
    assert len(slugify(long)) == 80
