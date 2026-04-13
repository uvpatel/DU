import pandas as pd

df = pd.read_json("hf://datasets/seansullivan/Next-JS-Docs/next-js-docs.json")

df.to_csv("tests/.csv", index=False)