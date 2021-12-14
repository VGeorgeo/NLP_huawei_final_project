def specific_cleaning(dataframe):
    '''
    Removing misleading rows based 
    on preliminary data analysis.
    
    It is specific for used database.
    '''
    
    #using data only with url
    dataframe = dataframe[dataframe['url'].notna()]
    
    #data without title is misleading
    dataframe = dataframe[dataframe['title'].notna()]
    
    #delete data withou russian and english words
    dataframe = dataframe[dataframe['title'].apply(lambda x: bool(re.search('[a-zA-Zа-яА-ЯёЁ]',x))
                                                   if type(x)==str else True)]
    indexes = dataframe['description'].apply(lambda x: bool(re.search('[a-zA-Zа-яА-ЯёЁ]',x))
                                                   if type(x)==str else True)
    dataframe['description'] = [tmp if zmp==True else np.nan for tmp,zmp in zip(dataframe['description'],
                                                                                indexes)]
    
    #specific cleaning
    del_index = np.load('../data/del_index')
    dataframe = dataframe.drop(index=list(set(del_index)))
    dataframe.loc[dataframe[dataframe['description']=='нет'].index,'description'] =        dataframe.loc[dataframe[dataframe['description']=='нет'].index,'description'].replace({"нет": np.nan})
    dataframe.loc[dataframe[dataframe['description']=='Это аннотация'].index,'description'] =        dataframe.loc[dataframe[dataframe['description']=='Это аннотация'].index,'description'].replace({"Это аннотация": np.nan}) 
    dataframe.loc[218,'title'] = 'Введение в искусственный интеллект'
    
    del_ind = dataframe[dataframe['description'].apply(lambda x: len(x)<21 if type(x)==str else True)][dataframe['description'].notna()].index
    del_ind.remove(59493)
    dataframe.loc[del_ind,'description'] = np.nan
    
    return dataframe


def html_cleaning(text):
    '''
    Delete most frequent html symbols
    '''
    
    rule = re.compile('<.*?>')
    text = re.sub(rule, ' ', text)
    rule = re.compile('&lt;.*?&gt;')
    text = re.sub(rule, ' ', text)
    text = re.sub('<p>', ' ', text)
    text = re.sub('</p>', ' ', text)
    text = re.sub(r' http\S+', ' ', text)
    text = re.sub(r' https\S+', ' ', text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = text.replace('&nbsp;', ' ')
    text = re.sub('<em>', ' ', text)
    text = re.sub('\xa0', ' ', text)
    text = re.sub('\t', ' ', text)
    text = re.sub('<b>', ' ', text)
    text = re.sub('</b>', ' ', text)
    text = re.sub('</em>', ' ', text)
    text = re.sub('%2f', ' ', text)
    text = re.sub('%3a', ' ', text)
    text = re.sub('<strong>', ' ', text)
    text = re.sub('</strong>', ' ', text)
    text = re.sub('<wbr>', ' ', text)
    text = re.sub('</wbr>', ' ', text)
    text = re.sub('&lt;', ' ', text)
    text = re.sub('»', ' ', text)
    text = re.sub('«', ' ', text)
    text = re.sub('#', ' ', text)
    text = re.sub('_', ' ', text)
    text = re.sub('\u202f', ' ', text)
    text = text.replace('ё', 'е')
    text = text.replace('Ё', 'Е')
    text = text.replace('    ', ' ')
    text = text.replace('   ', ' ')
    text = text.replace('  ', ' ')
    text = text.strip()
    
    return text


def add_punctuation(data_row):
    '''
    Add dot at the end of the text if there are no punctuation marks in it
    '''
    
    result = string.punctuation 
    data_row = data_row.apply(lambda x: ' ' + x if type(x)==str else '')
    
    return data_row


def preprocessing(text, low=True, punct=True, stopw=True, lemm=True, stemm=True):
    '''
    Text preprocessing: lowing, lemmatization, stemming
    :param low: if True make all letters low
    :param punct: if True make all punctuations from text
    :param stopw: if True delete all russian and englist stop words
    :param lemm: if True makes lemmatization of each word
    :param stemm: if True makes stemming of each russian word
    '''
    
    stopword_list = stopwords.words('english') + stopwords.words('russian')
    punct_list = list(string.punctuation)
    punct_list.remove('-')
    punct_list = str(punct_list)
    punct_list_add = '``“”—'
    morph = pymorphy2.MorphAnalyzer()
    snowball = SnowballStemmer(language='russian')
    

    if punct==True: text = text.translate(str.maketrans(punct_list, ' '*len(punct_list)))
    word_list = [tmp for tmp in word_tokenize(text)]  
    if punct==True: word_list = [tmp for tmp in word_list if tmp not in string.punctuation]
    if punct==True: word_list = [tmp for tmp in word_list if tmp not in punct_list_add]     
    if stopw==True: word_list = [tmp for tmp in word_list if tmp not in stopword_list]   
    if lemm==True: word_list = [morph.parse(tmp)[0][2] for tmp in word_list]
    if stemm==True: word_list = [snowball.stem(tmp) for tmp in word_list]
        
    text = " ".join(word_list)
    if low==True: text = text.lower()
        
    return text