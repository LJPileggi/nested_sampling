import os

from datetime import datetime

def file_encod(args):
    now = datetime.now()
    date = str(datetime.date(now))
    hour = str(datetime.time(now))
    hour = hour[:2] + hour[3:5] + hour[6:8]
    if args.automatised & args.X_stoch:
        output_path = os.path.abspath('./output/classical/'+f'X_stoch')
    elif args.automatised & args.trapezoid:
        output_path = os.path.abspath('./output/classical/'+f'trapezoid')
    elif args.automatised:
        output_path = os.path.abspath('./output/classical/'+f'normal')
    elif args.X_stoch:
        output_path = os.path.abspath('./output/classical/'+f'X_stoch')
    elif args.trapezoid:
        output_path = os.path.abspath('./output/classical/'+f'trapezoid')
    else:
        output_path = os.path.abspath('./output/classical/'+f'normal')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    out = os.path.join(output_path, f'{date}_{hour}.csv')
    return out
