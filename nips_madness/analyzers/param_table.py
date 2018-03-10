from ..utils import iteritemsdeep


def non_unique_columns(df):
    if len(df) == 0:
        return []

    non_unique = []
    for name, column in df.iteritems():
        if not all(column[0] == x for x in column[1:]):
            non_unique.append(name)
    return non_unique


def make_param_table(records, exclude_unique=True):
    import pandas
    rows = []
    for rec in records:
        rec_id = str(rec.datastore.directory)
        dct = {'.'.join(k): v for (k, v) in iteritemsdeep(rec.rc.dict)}
        dct['rec_id'] = rec_id
        rows.append(dct)
    param_table = pandas.DataFrame(rows)
    if exclude_unique:
        param_table = param_table[non_unique_columns(param_table)]
    return param_table


def merge_param_table(records, df):
    param_table = make_param_table(records)
    return df.merge(param_table, on='rec_id')
