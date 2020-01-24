import pickle

XML_DATA_FILE = 'data/drugs_pl.xml'
PICKLE_DATA_FILE = 'data/drug_names.p'


def gen_names_from_file(file=XML_DATA_FILE):
    prefix = 'nazwaProduktu="'
    suffix = '"'

    with open(file) as f:
        for line in f:
            start = line.find(prefix)
            if start == -1:
                continue
            start += len(prefix)

            end = line.find(suffix, start)
            name = line[start:end]
            yield name


def get_names():
    drug_names_gen = gen_names_from_file()
    drug_names = list(drug_names_gen)
    dedup_drug_names = list(set(drug_names))
    lower_names = [name.lower() for name in dedup_drug_names]
    return lower_names


def save_names(filename=PICKLE_DATA_FILE):
    drug_names = get_names()
    pickle.dump(drug_names, open(filename, "wb"))


def load_names(filename=PICKLE_DATA_FILE):
    drug_names = pickle.load(open(filename, "rb"))
    return drug_names


if __name__ == '__main__':
    save_names()
