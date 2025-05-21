import arxiv
from structures.Objects import text_data

class ArXivMiner:

    client = arxiv.Client()

    def __init__(self, keyword = "test"):
        self.keyWordSet = set()
        self.keyWordSet.add(keyword)
        self.usedKeyWordSet = set()

    def getArticles(self, amount: int, query: str) -> list[arxiv.Result]:
        search = arxiv.Search(
            query = query,
            max_results=amount
        )
        content_list = list()

        for result in self.client.results(search):
            content_list.append(result)
        
        return content_list
    
    def get_random_textdata_with_unique_kewords(self, batch_size) -> list[text_data]:
        new_keyword = " "
        
        result = []

        unusedKewordList = list(self.keyWordSet.difference(self.usedKeyWordSet))
        if len(unusedKewordList) > 0:
            new_keyword = unusedKewordList[0]

        articles = self.getArticles(batch_size, new_keyword)

        for article in articles:
            try:
                td = text_data(
                    title=article.title,
                    text=article.summary,
                    keywords=article.categories
                )
                result.append(td)

                for keyword in article.categories:
                    self.keyWordSet.add(keyword)
            except:
                pass

        return result