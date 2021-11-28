import json

with open("morph_params.json") as f:
    data = json.load(f)

print(f"The images are {data['Input files']}")
corrs = data["Correspondences"]
print(f"There are {len(corrs)} sets of correspondences")
for i in range(len(corrs)):
    print(f"There are {len(corrs[i][0][1])} correspdondences between image {corrs[i][0][0]} and image {corrs[i][1][0]}")
print(f"The output file is {data['Output file']}")
