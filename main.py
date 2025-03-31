from Classes import R1Z1


def r1z1():
    path_data = "6/r1z1.csv"
    Main_Class = R1Z1(path_data)
    Main_Class.load_Histogram("images/HISTOGRAM.png")
    Main_Class.load_EBF("images/EBF.png")
    Main_Class.print_stat()


if __name__ == "__main__":
    r1z1()
