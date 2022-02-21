import os
import csv
import argparse
import re

def parser():
    parser = argparse.ArgumentParser(description='Converts bash output into csv')
    parser.add_argument('--filename', metavar='filename', help='name of the file to be converted')
    args = parser.parse_args()
    return args

def reader(filename):
    n_points = [5, 10, 20, 35, 50, 75,
    100, 200, 350, 500, 750, 1000, 2000,
    3500, 5000, 7500, 10000, 20000, 35000]
    Z_dict = {point: [] for point in n_points}
    folder = './output/'
    with open(folder+filename, 'r') as f:
        start = 'points: '
        start1 = 'evidence: '
        end = ';'
        for i, line in enumerate(f):
            if i%4 == 0:
                n = re.search(start+'(.*)'+end, line)
                point = int(n.group(1))
            if i%4 == 2:
                z = re.search(start1+'(.*)'+end, line)
                evid = float(z.group(1))
                Z_dict[point].append(evid)
    output_path = os.path.abspath('output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    out = os.path.join(output_path, f'results.csv')
    with open(out, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for point in n_points:
            for value in Z_dict[point]:
                writer.writerow([point, point*90, value])

if __name__ == '__main__':
    args = parser()
    reader(args.filename)
