def tfidf_fit_save(data_row, name='model', ngram_range=(1, 2)):
    '''
    Make Tfidf model and save 
    :param ngram_range: size of ngram
    :param name: name of the file with model
    '''   
    
    model = TfidfVectorizer(ngram_range=ngram_range, stop_words='english', lowercase = False,)
    #model = TfidfVectorizer(ngram_range=ngram_range, stop_words='english', lowercase = False, tokenizer=word_tokenize)
    tfidf_matrix = model.fit_transform(data_row)
    
    file = open('../models/tf-idf/' + name + '.obj',"wb")
    pickle.dump(model,file)
    file.close()
    
    file = open('../models/tf-idf/matrix_' + name + '.obj',"wb")
    pickle.dump(tfidf_matrix,file)
    file.close()


def tfidf_load_model(name='model'):
    '''
    Make Tfidf matrix 
    :param low: size of ngram
    '''   
    
    file = open('../models/tf-idf/' + name + '.obj','rb')
    model = pickle.load(file)
    file.close()
    
    file = open('../models/tf-idf/matrix_' + name + '.obj','rb')
    tfidf_matrix = pickle.load(file)
    file.close()
    
    return model, tfidf_matrix    


def ruBERT_download_save():
    '''
    Download and save some RuBERT pretrained models
    '''  
    
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
    model = AutoModel.from_pretrained("cointegrated/rubert-tiny")
    file = open('../models/pretrained/RuBERT/ruBERT_tiny_tokenizer.obj',"wb")
    pickle.dump(tokenizer,file)
    file.close()
    file = open('../models/pretrained/RuBERT/ruBERT_tiny_model.obj',"wb")
    pickle.dump(model,file)
    file.close()
     
    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")
    file = open('../models/pretrained/RuBERT/ruBERT_base_tokenizer.obj',"wb")
    pickle.dump(tokenizer,file)
    file.close()
    file = open('../models/pretrained/RuBERT/ruBERT_base_model.obj',"wb")
    pickle.dump(model,file)
    file.close()
    
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-base-cased-nli-threeway")
    model = AutoModel.from_pretrained("cointegrated/rubert-base-cased-nli-threeway")
    file = open('../models/pretrained/RuBERT/ruBERT_threeway_tokenizer.obj',"wb")
    pickle.dump(tokenizer,file)
    file.close()
    file = open('../models/pretrained/RuBERT/ruBERT_threeway_model.obj',"wb")
    pickle.dump(model,file)
    file.close()
    
    model = AutoModel.from_pretrained('cointegrated/rubert-tiny-bilingual-nli')
    tokenizer = AutoTokenizer.from_pretrained('cointegrated/rubert-tiny-bilingual-nli')
    file = open('../models/pretrained/RuBERT/ruBERT_tiny_bilingual_tokenizer.obj',"wb")
    pickle.dump(tokenizer,file)
    file.close()
    file = open('../models/pretrained/RuBERT/ruBERT_tiny_bilingual_model.obj',"wb")
    pickle.dump(model,file)
    file.close()
    
    model = SentenceTransformer('sentence-transformers/LaBSE')
    file = open('../models/pretrained/LaBSE.obj',"wb")
    pickle.dump(model,file)
    file.close()


def ruBERT_load(name='ruBERT_tiny'):
    '''
    Load RuBERT pretrained model
    :param name: name of model
    '''   
    
    file = open('../models/pretrained/RuBERT/' + name + '_model.obj','rb')
    model = pickle.load(file)
    file.close()
    
    file = open('../models/pretrained/RuBERT/' + name + '_tokenizer.obj','rb')
    tokenizer = pickle.load(file)
    file.close()
    
    
    return model, tokenizer  


def LaBSE_load():
    '''
    Load LaBSE pretrained model
    :param name: name of model
    '''   
    
    file = open('../models/pretrained/LaBSE.obj','rb')
    model = pickle.load(file)
    file.close()
    
    return model 


def make_embeddings(text, model, tokenizer):
    '''
    Make embeddings for choosen model
    ''' 
    
    word_seq = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in word_seq.items()})
        
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    
    return embeddings[0].cpu().numpy()
