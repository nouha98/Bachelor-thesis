import pandas as pd 
import pickle
import unidecode, ast
import numpy as np

from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import Counter
import nltk
import string
import re

import config

import scraping_VF

################## classes 
class MeanEmbeddingVectorizer(object):
    def __init__(self, word_model):
        self.word_model = word_model
        self.vector_size = word_model.wv.vector_size

    def fit(self):  # comply with scikit-learn transformer requirement
        return self

    def transform(self, docs):  # comply with scikit-learn transformer requirement
        doc_word_vector = self.word_average_list(docs)
        return doc_word_vector

    def word_average(self, sent):

        mean = []
        for word in sent:
            if word in self.word_model.wv.index_to_key:
                mean.append(self.word_model.wv.get_vector(word))

        if not mean:  # empty words
            # If a text is empty, return a vector of zeros.
            # logging.warning(
            #     "cannot compute average owing to no vector for {}".format(sent)
            # )
            return np.zeros(self.vector_size)
        else:
            mean = np.array(mean).mean(axis=0)
            return mean

    def word_average_list(self, docs):

        return np.vstack([self.word_average(sent) for sent in docs])

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word_model):

        self.word_model = word_model
        self.word_idf_weight = None
        self.vector_size = word_model.wv.vector_size

    def fit(self, docs):  # comply with scikit-learn transformer requirement

        text_docs = []
        for doc in docs:
            text_docs.append(" ".join(doc))

        tfidf = TfidfVectorizer()
        tfidf.fit(text_docs)  # must be list of text string

        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)  # used as default value for defaultdict
        self.word_idf_weight = defaultdict(
            lambda: max_idf,
            [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()],
        )
        return self

    def transform(self, docs):  # comply with scikit-learn transformer requirement
        doc_word_vector = self.word_average_list(docs)
        return doc_word_vector

    def word_average(self, sent):

        mean = []
        for word in sent:
            if word in self.word_model.wv.index_to_key:
                mean.append(
                    self.word_model.wv.get_vector(word) * self.word_idf_weight[word]
                )  # idf weighted

        if not mean:  # empty words
            return np.zeros(self.vector_size)
        else:
            mean = np.array(mean).mean(axis=0)
            return mean

    def word_average_list(self, docs):
        return np.vstack([self.word_average(sent) for sent in docs])        
########### Function ####################
def ingredient_parser(ingreds):

    measures = [
        "teaspoon","t","tsp.","tablespoon","T",
        "tbl.","tb","tbsp.","fluid ounce","fl oz","gill","cup","c","pint",
        "p","pt","fl pt","quart","q","qt","fl qt","gallon","g","gal",
        "ml","milliliter","millilitre","cc","mL","l","liter","litre",
        "L","dl","deciliter","bulb","level","heaped","rounded",
        "whole","pinch","medium","slice","pound",
        "lb","#","ounce","oz","mg","milligram","milligramme",
        "g","gram","gramme","kg","kilogram","kilogramme","x","of","mm",
        "millimetre","millimeter","cm",
        "centimeter","m","meter","metre","inch",
        "in","milli","centi","deci","hecto","kilo",
    ]

    words_to_remove = ["advertisement", "advertisements",
                         "cup", "cups",
                         "tablespoon", "tablespoons", 
                         "teaspoon", "teaspoons", 
                         "ounce", "ounces",
                         "salt", 
                         "pound", "pounds","bunch","cloves","warm",
                       "white", "brown", "package", "box", "to taste"
                        "lowfat", "light", "shredded", "sliced", "all purpose", "all natural", "natural", "original", 
                        "gourmet", "traditional", "boneless", "skinless", "fresh", "nonfat", "pitted", "quick cooking", 
                        "unbleached", "part skim", "skim", "quickcooking", "oven ready", "homemade", "instant", "small", 
                        "extra large", "large", "chopped", "grated", "cooked", "stone ground", "freshly ground", 
                        "ground", "pure", "peeled", "deveined", "organic", "cracked", "granulated", "inch thick", 
                        "extra firm", "crushed", "flakes", "self rising", "diced", "crumbles", "crumbled", 
                        "whole wheat", "whole grain", "baby", "medium", "plain", "of", "thick cut", "cubed", "coarse", 
                        "free range", "seasoned", "canned", "multipurpose", "vegan", "thawed", "squeezed", 
                        "vegetarian", "fine", "zesty", "halves", "firmly packed", "drain", "drained","canned", "washed","smoked", ## added
                        "fresh", "minced", "chopped" ,"oil", "a","red","bunch","and", "clove", "or",
                        "leaf","large","extra","sprig","ground","handful",
                        "free", "small", "virgin", "range", "from", "dried","sustainable",
                        "black","peeled", "higher", "welfare", "seed", "for", "finely","freshly", "sea",
                        "quality", "white", "ripe", "few","piece","source", "to","organic", "flat",
                        "smoked", "ginger", "sliced", "green","picked", "the", "stick", "plain",
                        "plus", "mixed", "mint", "bay", "basil", "your", "cumin","optional",
                        "fennel","serve","mustard","unsalted","baby",
                        "fat","ask","natural","skin","roughly","into","such","cut",
                        "good","brown","grated","trimmed","oregano","powder",
                        "yellow","dusting","knob","frozen","on","deseeded",
                        "low","runny","balsamic","cooked","streaky","nutmeg","sage",
                        "rasher","zest", "pin","groundnut","breadcrumb",
                        "turmeric", "halved","grating","stalk","light","tinned","dry","soft",
                        "rocket","bone","colour","washed","skinless","leftover","splash","removed",
                        "dijon","thick","big","hot","drained","sized","chestnut","watercress",
                        "fishmonger","english","dill","caper","raw","worcestershire","flake","cider",
                        "cayenne","tbsp","leg","pine","wild","if","fine","herb","almond","shoulder",
                        "cube","dressing","with","chunk","spice","thumb","garam","new","little",
                        "punnet","peppercorn","shelled","saffron","other", "chopped","salt","olive","taste","can","sauce",
                        "water","diced","package","italian","shredded","divided","parsley","vinegar","all",
                        "purpose","crushed","juice","more","coriander","bell","needed","thinly","boneless",
                        "half","thyme","cubed","cinnamon","cilantro","jar","seasoning","rosemary","extract",
                        "sweet","baking","beaten","heavy","seeded","tin","vanilla","uncooked","crumb","style",
                        "thin","nut","coarsely","spring","chili","cornstarch","strip","cardamom","rinsed",
                        "root","quartered","head","softened","container","crumbled","frying", "lean",
                        "cooking","roasted","warm","whipping","thawed","corn","pitted","sun","kosher",
                        "bite","toasted","lasagna","split","melted","degree","lengthwise","romano","packed",
                        "pod","anchovy","rom","prepared","juiced","fluid","floret","room","active","seasoned","mix",
                        "deveined","lightly","anise","thai","size","unsweetened","torn","wedge","sour","basmati",
                        "marinara","dark","temperature","garnish","bouillon","loaf","shell","reggiano",
                        "canola","round","canned","ghee","crust","long","broken","ketchup","bulk","cleaned",
                        "condensed","sherry","provolone","cold","soda","cottage","spray","tamarind",
                        "pecorino","shortening","part","bottle","sodium","cocoa","grain","french","roast","stem","link","firm","asafoetida","mild",
                        "dash","boiling","oil","chopped","vegetable oil","chopped oil","garlic","skin off","bone out", "from sustrainable sources",                         
                         ]
    if isinstance(ingreds, list):
        ingredients = ingreds
    else:
        ingredients = ast.literal_eval(ingreds)

    translator = str.maketrans("", "", string.punctuation)
    lemmatizer = WordNetLemmatizer()
    ingred_list = []
    for i in ingredients:
        i.translate(translator)
        # We split up with hyphens as well as spaces
        items = re.split(" |-", i)
        # Get rid of words containing non alphabet letters
        items = [word for word in items if word.isalpha()]
        # Turn everything to lowercase
        items = [word.lower() for word in items]
        # remove accents
        items = [
            unidecode.unidecode(word) for word in items
        ]  #''.join((c for c in unicodedata.normalize('NFD', items) if unicodedata.category(c) != 'Mn'))
        # Lemmatize words so we can compare words to measuring words
        items = [lemmatizer.lemmatize(word) for word in items]
        # Gets rid of measuring words/phrases, e.g. heaped teaspoon
        items = [word for word in items if word not in measures]
        # Get rid of common easy words
        items = [word for word in items if word not in words_to_remove]
        if items:
            ingred_list.append(" ".join(items))
    return ingred_list

# neaten the ingredients being outputted 
def ingredient_parser_final(ingredient):
    if isinstance(ingredient, list):
        ingredients = ingredient
    else:
        ingredients = ast.literal_eval(ingredient)
    
    ingredients = ','.join(ingredients)
    ingredients = unidecode.unidecode(ingredients)
    return ingredients

def title_parser(title):
    title = unidecode.unidecode(title)
    return title 

# get corpus with the documents sorted in alphabetical order
def get_and_sort_corpus(data):
    corpus_sorted = []
    for doc in data.ingredients_parsed.values:
        #doc.sort()
        corpus_sorted.append(doc)
    return corpus_sorted

def process_recipes(data):
    # list of stopwords
    stop_word_list = nltk.corpus.stopwords.words("english")

    data["ingredients_parsed"] = data["ingredients"].apply(
          lambda x: ingredient_parser(x.split(";")))
    return data


######## Data 2 df_recipes with  web recipe scrapping ##################
def get_recommendations(N, scores):
    """
    Top-N recomendations order by score
    """
    # load in recipe dataset    
    data = pd.read_csv(config.RECIPES1)

    # order the scores with and filter to get the highest N scores
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N]

    # create dataframe to load in recommendations
    recommendation = pd.DataFrame(columns=["recipe", "ingredients", "score", "instructions","img_url"])
    count = 0

    for i in top:

        url = data["recipe_urls"][i]
        recipe_name, ingredients, instructions, img_URL = scraping_VF.R_scrape(url)
        recommendation.at[count, "recipe"] = title_parser(recipe_name)

        recommendation.at[count, "ingredients"] = ingredients #ingredient_parser_final(ingredients)
        recommendation.at[count, "instructions"] = instructions
        recommendation.at[count, "img_url"] = img_URL
        recommendation.at[count, "score"] = f"{scores[i]}"
        count += 1
    #print(scores[i])    
    return recommendation

def get_recs(ingredients, N=5, mean=False):
    
    # load in word2vec model
    model = Word2Vec.load(config.recipe_model1)
    model.init_sims(replace=True)

    if model:
        print("Successfully loaded model")

    # load in data

    #process data + parse ingredients
    data = pd.read_csv(config.RECIPES1)

    #data["ingredients_parsed"] = data.ingredients.apply(ingredient_parser)
    data = process_recipes(data)

    #data = process_recipes(data)
    #data = data.dropna()

    # create corpus
    corpus = get_and_sort_corpus(data)

    if mean:
        # get average embdeddings for each document
        mean_vec_tr = MeanEmbeddingVectorizer(model)
        doc_vec = mean_vec_tr.transform(corpus)
        doc_vec = [doc.reshape(1, -1) for doc in doc_vec]
        assert len(doc_vec) == len(corpus)
    else:
        # use TF-IDF as weights for each word embedding
        tfidf_vec_tr = TfidfEmbeddingVectorizer(model)
        tfidf_vec_tr.fit(corpus)
        doc_vec = tfidf_vec_tr.transform(corpus)
        doc_vec = [doc.reshape(1, -1) for doc in doc_vec]
        assert len(doc_vec) == len(corpus)

    # create embessing for input text
    input = ingredients
    # create tokens with elements
    input = input.split(",")
    # parse ingredient list
    input = ingredient_parser(input)
    # get embeddings for ingredient doc
    if mean:
        input_embedding = mean_vec_tr.transform([input])[0].reshape(1, -1)
    else:
        input_embedding = tfidf_vec_tr.transform([input])[0].reshape(1, -1)

    # get cosine similarity between input embedding and all the document embeddings
    cos_sim = map(lambda x: cosine_similarity(input_embedding, x)[0][0], doc_vec)
    scores = list(cos_sim)
    # Filter top N recommendations
    recommendations = get_recommendations(N, scores)
    return recommendations

##########  another approach### 
def RecSys(ingredients, N=5):

    # load in tdidf model and encodings 
    with open('tfidf_encodings.pkl', 'rb') as f:
        tfidf_encodings = pickle.load(f)

    with open('tfidf.pkl', "rb") as f:
        tfidf = pickle.load(f)

    # parse the ingredients using my ingredient_parser 
    try: 
        ingredients_parsed = ingredient_parser(ingredients)
    except:
        ingredients_parsed = ingredient_parser([ingredients])
    
    ingredients_parsed = " ".join(ingredients_parsed)
    # use our pretrained tfidf model to encode our input ingredients
    ingredients_tfidf = tfidf.transform([ingredients_parsed])

    # calculate cosine similarity between actual recipe ingreds and test ingreds
    cos_sim = map(lambda x: cosine_similarity(ingredients_tfidf, x), tfidf_encodings)
    scores = list(cos_sim)

    # Filter top N recommendations 
    recommendations = get_recommendations(N, scores)
    return recommendations

if __name__ == "__main__":
    # test ingredients
    test_ingredients = "Apple, tomato, Pepper, kiwi "
    #test_ingredients = test_ingredients.split(",")
    #recs = RecSys(test_ingredients)
    recs = get_recs(test_ingredients)
    recs1 = get_recs(test_ingredients,mean=True)
    print(recs)
    print("TFID",recs.score)
    print(recs1)
    print("Mean Embed",recs1.score)