from pymed import PubMed
import itertools
from structures.Objects import text_data

class PubMedMiner:
    
    
    #TODO toolname, email
    pubmed: PubMed
    keyWordSet: set
    usedKeyWordSet: set

    def __init__(self, keyword = "test"):
        self.keyWordSet = set()
        self.keyWordSet.add(keyword)
        self.pubmed = PubMed()
        self.usedKeyWordSet = set()


    def getArticles(self, amount, query:str) -> itertools.chain:
        try:
            results: itertools.chain = self.pubmed.query(query=query, max_results=amount)
        except:
            return itertools.chain()
        return results
    
    def get_random_textdata_with_unique_kewords(self, batch_size) -> list[text_data]:
        new_keyword = " "
        
        result = []

        unusedKewordList = list(self.keyWordSet.difference(self.usedKeyWordSet))
        if len(unusedKewordList) > 0:
            new_keyword = unusedKewordList[0]

        articles = self.getArticles(batch_size, "(('" + new_keyword + "'[MeSH Terms]) OR ('" + new_keyword + "'[Other Term])) AND (english[Language])")

        for article in articles:
            try:
                td = text_data(
                    title=article.title,
                    text=article.abstract,
                    keywords=article.keywords
                )
                result.append(td)

                for keyword in article.keywords:
                    self.keyWordSet.add(keyword)
            except:
                pass

        return result
        
