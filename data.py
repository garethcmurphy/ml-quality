#!/usr/bin/env python3
"""generate data"""
import random

class Data():
    """generate data"""
    def __init__(self):
        pass

    def generate(self):
        """generate data"""
        for i in range(1, 10):
            pid = random.randint( 1e5, 9e5)
            hindex = random.randint(0, 100)
            data_type =  random.choice( ['test', 'production'] )

            print("{}, {}, {}".format(pid, hindex, data_type))



def main():
    data = Data()
    data.generate()


if __name__ == "__main__":
    main()