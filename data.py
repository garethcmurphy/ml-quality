#!/usr/bin/env python3
"""generate data"""
import random


class Data():
    """generate data"""
    file_name = "./data.csv"

    def __init__(self):
        pass

    def generate(self):
        """generate data"""
        with open(self.file_name, "w") as file:
            print("pid,hindex,data_type,target", file=file)
            for _ in range(1, 1001):
                pid = random.randint(1e5, 9e5)
                hindex = random.randint(0, 100)
                data_type = random.choice(['test', 'production'])
                target = random.randint(0, 1)
                print("{},{},{},{}".format(pid, hindex, data_type, target), file=file)


def main():
    """main"""
    data = Data()
    data.generate()


if __name__ == "__main__":
    main()
