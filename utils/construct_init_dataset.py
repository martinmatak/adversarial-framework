import sys

#csv_path = sys.argv[1]
num_of_categories = 3

age_span_per_category = int(101/int(num_of_categories))
print("age span per category: ", age_span_per_category)

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
        for age in (current_age, current_age + age_span_per_category):
            mapping_age_to_category[age] = category
    else:
        print("category " + str(category) + ": " + str(current_age) + " - " + str(max_age))
        for age in (current_age, max_age + 1):
            mapping_age_to_category[age] = category
        break

    current_age += age_span_per_category
    category += 1


