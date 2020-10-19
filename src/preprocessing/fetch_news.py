"""
Program fetch news from web.
@Author: namtran.ase@gmail.com.
"""
from newsplease import NewsPlease

def main():
    """Main program.
    """
    url = "https://vnexpress.net/the-gioi"
    article = NewsPlease.from_url(url)
    print(article.title)

if __name__ == "__main__":
    main()