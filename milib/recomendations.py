def make_tfidf_recomendation(all_embeddings, text_embedding, model, data, data_orig, number=10):
    '''
    Compare embeddings with cosine similarity and choose the 10 most similar
    ''' 

    tfidf_text = model.transform([text_embedding])
    similarity = cosine_similarity(all_embeddings, tfidf_text).reshape(-1)
    indices = (-similarity).argsort()[:10]
    indices = data.iloc[indices,0]
    if max(similarity)<0.3:
      return False
    else:
      data_rec = data_orig.iloc[indices,1:3]
      return data_rec.reset_index(drop=True)


def make_recomendation(all_embeddings, text_embedding, model, tokenizer, data, data_orig, number=10):
    '''
    Compare embeddings with cosine similarity and choose the 10 most similar
    ''' 

    text_embedding = make_embeddings(text_embedding, model, tokenizer)
    similarity = cosine_similarity(all_embeddings, [text_embedding]).reshape(-1)
    indices = (-similarity).argsort()[:10]
    indices = data.iloc[indices,0]
    if max(similarity)<0.3:
      return False
    else:
      data_rec = data_orig.iloc[indices,1:3]
      return data_rec.reset_index(drop=True)


def make_labse_recomendation(all_embeddings, text_embedding, model, data, data_orig, number=10):
    '''
    Compare embeddings with cosine similarity and choose the 10 most similar
    ''' 

    text_embedding = model.encode(text_embedding)
    similarity = cosine_similarity(all_embeddings, [text_embedding]).reshape(-1)
    indices = (-similarity).argsort()[:number]
    indices = data.iloc[indices,0]
    if max(similarity)<0.3:
      return False
    else:
      data_rec = data_orig.iloc[indices,1:3]
      return data_rec.reset_index(drop=True)