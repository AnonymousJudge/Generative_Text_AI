import wikipedia

from structures.Objects import text_data

class WikipediaMiner:

    def __init__(self):
        wikipedia.set_rate_limiting(False)

    def __getRandomTextData(self) -> text_data:
        data = None
        title = wikipedia.random()
        try: 
            article = wikipedia.page(title=title)
            data = text_data(title=title,
                             text = article.content,
                             keywords = article.categories)
        except:
            pass

        return data
    
    def getRandomTextData(self) -> text_data:
        data = None
        while data is None:
            data = self.__getRandomTextData()
        return data
