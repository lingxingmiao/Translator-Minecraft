from TranslatorLib import pickle, Mods

def load(d):
    index = Mods().索引类(d["模式"])()
    for k, v in d.items():
        setattr(index, k, v)
    return index
def read_index(filename: str):
    with open(filename, 'rb') as f:
        d = pickle.load(f)
    return load(d)
def write_index(index, filename: str):
    index.save(filename)
def index_cpu_to_gpu(index, **kwargs):
    index.gpu()
    return index
def index_gpu_to_cpu(index, **kwargs):
    index.cpu()
    return index
def serialize_index(index):
    return index.serialize()
def deserialize_index(data):
    if isinstance(data, bytes):
        return load(pickle.loads(data))
    return load(pickle.load(data))
