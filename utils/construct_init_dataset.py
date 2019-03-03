import pandas as pd
import random as rng

csv_path = '/Users/mmatak/dev/thesis/datasets/appa-real-release/gt_avg_test.csv'
csv_path = '/root/datasets/appa-real-release/gt_avg_test.csv'
num_of_categories = 5

age_span_per_category = int(101/int(num_of_categories))
print("age span per category: ", age_span_per_category)

total_dataset_size = 500
attack_dataset_size = 100
start_age = 0
max_age = 100
current_age = start_age

# initialize array
mapping_age_to_category = {}
for age in range(0,max_age+1):
    mapping_age_to_category[age] = -1

category = 0
while True:
    if current_age + 2 * age_span_per_category <= max_age:
        print("category " + str(category) + ": " + str(current_age) + " - " + str(current_age + age_span_per_category - 1))
        for age in range(current_age, current_age + age_span_per_category):
            mapping_age_to_category[age] = category
    else:
        print("category " + str(category) + ": " + str(current_age) + " - " + str(max_age))
        for age in range(current_age, max_age + 1):
            mapping_age_to_category[age] = category
        break

    current_age += age_span_per_category
    category += 1

print("total categories: ", str(category + 1))
max_samples_per_category = int(total_dataset_size / (category + 1))

num_of_samples_per_category = {}
for i in range(0, category + 1):
    num_of_samples_per_category[i] = 0

files_to_take = set()

df = pd.read_csv(str(csv_path))
total_samples = 0
for i, row in df.iterrows():
    age = min(max_age, int(row.apparent_age_avg))
    category = mapping_age_to_category[age]
    if num_of_samples_per_category[category] <= max_samples_per_category:
        num_of_samples_per_category[category] += 1
        files_to_take.add(str(row.file_name) + "," + str(age))
        total_samples += 1
    if total_samples == total_dataset_size:
        break

for category in num_of_samples_per_category.keys():
    print("Category: " + str(category) + ", num of samples: " + str(num_of_samples_per_category[category]))

print("Total number of samples: ", total_samples)

if False:
    print("total size is less than expected dataset size, filling it up now with other samples...")
    for i, row in df.iterrows():
        age = min(max_age, int(row.apparent_age_avg))
        if str(row.file_name) + "," + str(age) not in files_to_take:
            category = mapping_age_to_category[age]
            num_of_samples_per_category[category] += 1
            files_to_take.add(str(row.file_name) + "," + str(age))
            total_samples += 1
            if total_samples == total_dataset_size:
                break

for category in num_of_samples_per_category.keys():
    print("Category: " + str(category) + ", num of samples: " + str(num_of_samples_per_category[category]))

print("Total number of samples: ", total_samples)

print("file_name,apparent_age_avg", file=open("custom-dataset.csv", "w"))
for filename in files_to_take:
    print(filename, file=open("custom-dataset.csv", "a"))

print("constructing set of samples for attack now...")
current_size = 0
attack_set = set()
while size(attack_set) < attack_dataset_size:
    row = df[rng.randint(0, df.size())]
    file_age = str(row.file_name) + "," + str(age)
    # if in training dataset, skip sample
    if file_age in files_to_take:
        continue
    attack_set.add(file_age)

print("Constructed")
attack_filename = "attack-samples.csv"
print("file_name,apparent_age_avg", file=open(attack_filename, "w"))
for filename in attack_set:
    print(filename, file = open(attack_filename, "a"))
print("Attack samples stored in " + attack_filename)