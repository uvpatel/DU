import DU

print(DU.greet())

df = DU.load_csv("tests/data.csv")
print(DU.basic_stats(df))