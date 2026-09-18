try:
    import jieba
    from TranslatorLib import Mods

    @Mods().注册分词器("zh_cn")
    def Tokenizer(texts: list) -> list:
        return [list(jieba.cut(text)) for text in texts]
except: pass