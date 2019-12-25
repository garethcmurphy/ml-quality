#!/usr/bin/env python3

class Data():
    def __init__(self):
        pass

    def generate(self):
        for i  in range(1,10):
            print("{}, {}, {}".format(i,i,i))



def main():
    data = Data()
    data.generate()


if __name__ == "__main__":
    main()