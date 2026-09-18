try:
    from TranslatorLib import Mods, bm25s

    @Mods().注册分词器("en_us")
    def Tokenizer(texts: list) -> list:
        return bm25s.tokenize(texts)
except: pass