import os

from datetime import datetime

def file_encod(args, no_search):
    now = datetime.now()
    date = str(datetime.date(now))
    hour = str(datetime.time(now))
    hour = hour[:2] + hour[3:5] + hour[6:8]
    if no_search:
        output_path = os.path.abspath('./output/diffusive/normal')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        out = os.path.join(output_path, f'{date}_{hour}.csv')
    elif args.search_L_per_level:
        output_path = os.path.abspath('./output/diffusive/normal')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        out = os.path.join(output_path, f'{date}_{hour}.csv')
    elif args.search_lam:
        output_path = os.path.abspath('./output/diffusive/lambda')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        out = os.path.join(output_path, f'{date}_{hour}.csv')
    elif args.search_beta:
        output_path = os.path.abspath('./output/diffusive/beta')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        out = os.path.join(output_path, f'{date}_{hour}.csv')
    elif args.search_quantile:
        output_path = os.path.abspath('./output/diffusive/quantile')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        out = os.path.join(output_path, f'{date}_{hour}.csv')
    return out
