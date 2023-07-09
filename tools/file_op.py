import os
import sys

#sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
import pickle

def write_pkl(content, path, print_op=True):
    '''write content on path with path
    Dependency : pickle
    Args:
        content - object to be saved
        path - string
                ends with pkl
    '''
    with open(path, 'wb') as f:
        if print_op: print("Pickle is written on %s"%path)
        try: pickle.dump(content, f)
        except OverflowError: pickle.dump(content, f, protocol=4)

def read_pkl(path, encoding='ASCII', print_op=True):
    '''read path(pkl) and return files
    Dependency : pickle
    Args:
        path - string
               ends with pkl
    Return:
        pickle content
    '''
    if print_op: print("Pickle is read from %s"%path)
    with open(path, 'rb') as f: return pickle.load(f, encoding=encoding)

def create_dir(dirname, print_op=True):
   '''create directory named dirname
   Dependency : os
   Args:
       dirname - string
                 directory named
   '''
   if not os.path.exists(dirname):
       if print_op: print("Creating %s"%dirname)
       try:
           os.makedirs(dirname)
       except FileExistsError:
           pass
   else:
       if print_op: print("Already %s exists"%dirname)

def create_muldir(*args):
   for dirname in args: create_dir(dirname)
   
def write_txt(content, path, print_op=True):
    if print_op: print("Text is written on %s"%path)
    file = open(path,"w")
    file.write(content)
    file.close()
