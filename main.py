
from helper.miner.WikipediaMiner import WikipediaMiner
from helper.miner.PubMedMiner import PubMedMiner
from helper.miner.ArXivMiner import ArXivMiner
from tokenizer.Tokenizer import Tokenizer
from Embedder import Embedder
import sys
from multiprocessing import Process, Manager, Pool

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

different_tokens = 2000
wiki_miner = WikipediaMiner()
pubmed_miner = PubMedMiner()
arxiv_miner = ArXivMiner()


tokenizer_data_path = "./data/tokenizer/saves"

def get_text_data(a: int) -> list[str]:
    
    
    print("\tstarted pulling text data")
    wm = Manager()
    wiki_list = wm.list()
    
    pm = Manager()
    pubmed_list = pm.list()

    am = Manager()
    arxiv_list = am.list()

    wiki_p = Process(target=get_wiki_data, args=(a, wiki_list))
    pubmed_p = Process(target=get_pubmed_data, args=(a, pubmed_list))
    arxiv_p = Process(target=get_arxiv_data, args=(a, arxiv_list))

    wiki_p.start()
    pubmed_p.start()
    arxiv_p.start()

    wiki_p.join()
    pubmed_p.join()
    arxiv_p.join()

    print("\tfinished pulling text data")

    data = list(wiki_list) + list(pubmed_list) + list(arxiv_list)
    
    return data

def get_wiki_data(a:int, li:list):
    print("\t\tstarted wiki-miner")
    for _ in range(a):
        textData = wiki_miner.getRandomTextData()
        li.append(textData.text)
    print("\t\twiki-miner finished")
    return li

def get_pubmed_data(a:int, li:list):
    print("\t\tstarted pubmed-miner")
    batch_size: int = int(5)
    for _ in range(a):
        texts = pubmed_miner.get_random_textdata_with_unique_kewords(batch_size)
        for t in texts:
            li.append(t.text)
    print("\t\tpubmed-miner finished")
    return li

def get_arxiv_data(a:int, li:list):
    print("\t\tstarted arxiv-miner")
    batch_size: int = int(5)
    for _ in range(a):
        texts = arxiv_miner.get_random_textdata_with_unique_kewords(batch_size)
        for t in texts:
            li.append(t.text)
    print("\t\tarxiv-miner finished")
    return li

def train_tokenizer(a: int):
    print("started data preperation")
    text = get_text_data(a)
    print("finished data preperation")
    print("started training tokenizer")
    tokenizer = Tokenizer()
    tokenizer.train(text, different_tokens, True)
    print("finished training tokenizer")
    tokenizer.save(tokenizer_data_path)
    print("\tsaved data")

def train_embedder_neural_network(epochs:int, text_size:int, min:int, max:int, displayUpdate:int):
    print("started data preperation")
    embedder = Embedder(different_tokens)
    tokenizer = Tokenizer()
    tokenizer.load(tokenizer_data_path + ".model")
    tests = get_text_data(text_size)
    print("started encoding text_data")
    with Pool() as pool:
        e_texts = pool.map(tokenizer.encode, tests)
    print("finished encoding text_data")
    print("finished data preperation")
    print("started training embedder NN")
    embedder.train_neural_network(epochs=epochs, encoded_texts=e_texts, min_keys=min, max_keys=max, displayUpdate=displayUpdate)
    print("finished training embedder NN")


if __name__ == '__main__':
    train_tokenizer(1000)
    for _ in range(10):
        train_embedder_neural_network(epochs=100, text_size = 20, min=1, max=2, displayUpdate=10)
