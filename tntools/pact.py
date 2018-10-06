import hashlib
import pickle
import os
import logging
import yaml

""" A module for storing data on the disk. The idea is that every piece
of data is uniquely identified by a name for what it is, and a dictionary
of properties it has. Given data, a name and dictionary, Pact pickles
the data and stores in a given folder, and given a just a name and
dictionary Pact can fetch this data from the same folder. Each stored
file is also accompanied by a YAML file, that gives the dictionary keys
in a human-readable format.

Pact should also keep an index file that keeps tracks of all the data
that is currently stored, but implementing this is only half-done.

The datadispenser.py module makes extensive use of Pact as a storage
backend.
"""

# TODO Is the whole class structure necessary?
class Pact:
    def __init__(self, folder):
        self.folder = folder
        self.indexpath = folder + "pactindex.p"

    def generate_filename(self, name, d, extension=".p", **kwargs):
        d = self.update_dict(d, **kwargs)
        suffix = type(self).dict_to_string(d)
        suffix = type(self).hash_str(suffix)
        filename =  name + "_" + suffix + extension
        return filename

    def generate_path(self, *args, filename=None, **kwargs):
        if filename is None:
            filename = self.generate_filename(*args, **kwargs)
        path = self.folder + filename
        return path

    @staticmethod
    def dict_to_string(d):
        d = sorted(d.items())
        first_item = d[0]
        postfix = "{}-{}".format(first_item[0], first_item[1])
        for k, v in d[1:]:
            postfix += ",_{}-{}".format(k,v)
        return postfix

    @staticmethod
    def hash_str(string):
        bstring= string.encode('UTF-8')
        string = hashlib.md5(bstring).hexdigest()
        return string

# # #

    @classmethod
    def dict_to_hashable(cls, d):
        # TODO Could we use a yaml.dump string for this purpose?
        # For the purpose of using a dict d as a key for some other
        # dict, we convert d into something hashable.
        d = d.copy()
        for k, v in d.items():
            try:
                hash(v)
            except TypeError:
                if type(v) is list:
                    v = tuple(v)
                elif type(v) is dict:
                    v = cls.dict_to_hashable(v)
                else:
                    raise
                d[k] = v
        fs = frozenset(d.items())
        return fs

    def update_dict(self, d, **kwargs):
        res = d.copy()
        res.update(d)
        res.update(kwargs)
        return res

    def store(self, data, name, d, extension=".p", **kwargs):
        d = self.update_dict(d, **kwargs)
        filename = self.generate_filename(name, d, extension=extension)
        path = self.generate_path(filename=filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        self.write_to_index(filename, d)
        self.store_pars_file(name, d)
        logging.info("Wrote to {}".format(path))
        return

    def store_pars_file(self, name, d, **kwargs):
        d = self.update_dict(d, **kwargs)
        filename = self.generate_filename(name, d, extension=".yaml")
        path = self.generate_path(filename=filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(d, f, default_flow_style=False)
        return

    def write_to_index(self, filename, d):
        try:
            with open(self.indexpath, "rb") as f:
                index = pickle.load(f)
        # TODO Why do we need to catch EOFErrors as well? Why do they
        # sometimes get raised? Corrupt files?
        except (EOFError, FileNotFoundError):
            index = dict()
        fs = type(self).dict_to_hashable(d)
        index[fs] = filename
        with open(self.indexpath, "wb") as f:
            pickle.dump(index, f)
        return

    def fetch(self, name, d, extension=".p", **kwargs):
        d = self.update_dict(d, **kwargs)
        filename = self.generate_filename(name, d, extension=extension)
        path = self.generate_path(filename=filename)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        logging.info("Read from {}".format(path))
        return data

    def exists(self, name, d, extension=".p", **kwargs):
        d = self.update_dict(d, **kwargs)
        filename = self.generate_filename(name, d, extension=extension)
        path = self.generate_path(filename=filename)
        exists = os.path.isfile(path)
        return exists

    def reconstruct_index(self):
        # TODO
        pass

