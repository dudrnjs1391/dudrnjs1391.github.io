---
layout: post
categories: Multiprocessing
title: "Multiprocessing"
tags: [Python, Tokenizing, Multiprocessing]
---

Python에서 수많은 데이터를 Tokenizing하려면 엄청난 시간이 든다.<br>
NSMC json file을 불러와 진행하려다보니 너무 시간이 오래 걸려 Multi Processing을 도입해보았다.

    python code : 

    import os
    from multiprocessing import Pool
    from functools import partial
    import pandas as pd
    import numpy as np
    
    def twitter_morphs(doc):
        twitter = konlpy.tag.Okt()
        return twitter.morphs(doc)
    
    def tokenizing(data, tokenizer, token_col_nm):
        data['TOKEN_TEXT'] = data[token_col_nm].apply(tokenizer)
        print('pid :', os.getpid(), 'TEXT TOKENIZING..')
        return data
    
    def parallelize_dataframe(df, func, tokenizer, token_col_nm, num_cores):
        df_split = np.array_split(df, num_cores)
        time1 = time.time()
        pool = Pool(num_cores)
        df = pd.concat(pool.map(partial(func, tokenizer = tokenizer, token_col_nm = token_col_nm), df_split))
        pool.close()
        pool.join()
        print("Elapsed Time : ", round((time.time()-time1) /60, 3), "minutes.")
        return df
        
    num_cores = 16
    nsmc_df = parallelize_dataframe(nsmc_df, tokenizing, twitter_morphs, 'TEXT', num_cores)

- tokenizer는 일단 twitter로 진행
- tokenizing 이게 필요할까? 위의 함수로 적용할 수 있을듯
- Multi Processing은 Pool을 사용해보자.

- 원래 20분이 걸리던 tokenizing이 2분 30초가 걸렸다.