import pickle
import numpy as np

# 层级打印一个字典
def print_dict(dict_, content=False, level=0):
    for k, v in dict_.items():
        print('\t' * level + f'{k}', end=' ')
        if type(v) == dict:
            print('\r')
            print_dict(v, content, level + 1)
        elif content:
            try: 
                v = np.round(v, 1)
            except:
                pass
            print(v.shape) if isinstance(v, np.ndarray) else print('')
            print(v)
            print('-------------------------------------')
        else:
            print(v.shape) if isinstance(v, np.ndarray) else print('\r')