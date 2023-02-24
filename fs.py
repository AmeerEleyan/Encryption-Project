# Created by Ameer Eleyan
# at 2/17/2023 2:55 PM

def calculate_similarity(list1, list2, b_size):
    count = 0
    for b, a in zip(list1, list2):
        if a == b:
            count = count + 1
    return count / b_size


for i in range(0, 5, 2):
    print(i)


