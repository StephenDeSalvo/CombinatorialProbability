

import csv



def parse_full_path_filename(full_path_filename):

    if '.' in full_path_filename:
        splitter = full_path_filename.split('.')
        filename = '.'.join(splitter[:-1])
        extension = splitter[-1]
    else:
        filename = full_path_filename
        extension = ''
        

    splitter = filename.split('/')
    path = '/'.join(splitter[:-1])
    filename = splitter[-1]

    return path, filename, extension

def parse_dictionary_filename(filename):

    if '_to_' in filename:
        key_str, value_str = filename.split('_to_')
    else:
        key_str = filename
        value_str = ''

    if '_x_' in key_str:
        key_list = key_str.split('_x_')
    else:
        key_list = [] if key_str == '' else [key_str]

    if '_x_' in value_str:
        value_list = value_str.split('_x_')
    else:
        value_list = [] if value_str == '' else [value_str]


    
    return key_list, value_list


def reads(full_path_filename, **kwargs):

    path, filename, extension = parse_full_path_filename(full_path_filename)

    if 'key' not in kwargs and 'value' not in kwargs:
        key_list, value_list = parse_dictionary_filename(filename)
    elif 'value' not in kwargs and 'key' in kwargs:
        if isinstance(kwargs['key'], str):
            key_list = [kwargs['key']]
        else:
            key_list = kwargs['key']
    elif 'key' in kwargs and 'value' in kwargs:

        if isinstance(kwargs['key'], str):
            key_list = [kwargs['key']]
        else:
            key_list = kwargs['key']

        if isinstance(kwargs['value'], str):
            value_list = [kwargs['value']]
        else:
            value_list = kwargs['value']
    else:
        raise ValueError('NOT A VALID OPTION, recheck your input arguments')



    errors = 'replace' if 'errors' not in kwargs else kwargs['errors']
    delimiter = '|' if 'delimiter' not in kwargs else kwargs['delimiter']


    if len(value_list) == 0:
        mapping = []
    else:
        mapping = {}

    with open(full_path_filename, 'r', encoding='utf-8', errors=errors) as file:
        reader = csv.reader(file, delimiter=delimiter)

        header = next(reader)
        key_index = []
        for key in key_list:
            key_index.append(header.index(key))
        value_index = []
        for value in value_list:
            value_index.append(header.index(value))

        for row in reader:

            key = []
            for index in key_index:
                key.append(row[index])

            value = []
            for index in value_index:
                value.append(row[index])

            if len(key) == 1:
                key = key[0]

            if len(value) == 1:
                value = value[0]

            if len(value) == 0:
                mapping.append(key)
            else:
                mapping[key] = value


    return mapping




def writes(full_path_filename, mapping, **kwargs):

    path, filename, extension = parse_full_path_filename(full_path_filename)

    key_list, value_list = parse_dictionary_filename(filename)

    errors = 'replace' if 'errors' not in kwargs else kwargs['errors']
    delimiter = '|' if 'delimiter' not in kwargs else kwargs['delimiter']
    sort = False if 'sort' not in kwargs else kwargs['sort']

    with open(full_path_filename, 'w', encoding='utf-8', errors=errors) as file:

        file.write(delimiter.join(key_list))
        if len(value_list) > 0:
            file.write(delimiter+delimiter.join(value_list))
        file.write('\n')

        if len(value_list) > 0:

            if sort:
                if sort == 'natural':
                    mapping_items = sorted([(float(x), y) for x, y in mapping.items()])
                else:
                    mapping_items = sorted(mapping.items())
            else:
                mapping_items = mapping.items()

            if len(key_list) >= 2 and len(value_list) >= 2:
                for key, value in mapping_items:
                    key_str = delimiter.join(key)
                    value_str = delimiter.join(value)
                    file.write(key_str+delimiter+value_str+'\n')
            elif len(key_list) == 1 and len(value_list) >= 2:
                for key, value in mapping_items:
                    value_str = delimiter.join(value)
                    file.write(key+delimiter+value_str+'\n')
            elif len(key_list) >= 2 and len(value_list) == 1:
                for key, value in mapping_items:
                    key_str = delimiter.join(key)
                    file.write(key_str+delimiter+value+'\n')
            elif len(key_list) == 1 and len(value_list) == 1:
                for key, value in mapping_items:
                    file.write(key+delimiter+value+'\n')

        else:

            if sort:
                if sort == 'natural':
                    mapping_items = sorted([float(x) for x in mapping])
                else:
                    mapping_items = sorted(mapping)
            else:
                mapping_items = mapping

            if len(key_list) >= 2:
                for key in mapping_items:
                    key_str = delimiter.join(key)
                    file.write(key_str+'\n')
            elif len(key_list) == 1:
                for key in mapping_items:
                    file.write(key+'\n')

