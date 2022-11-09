import math

def euclidean_distance(x1, x2):
    return math.sqrt(sum([(x1_element - x2_element) ** 2 for (x1_element, x2_element) in zip(x1, x2)]))

def manhattan_distance(x1, x2):
    return sum([abs(x1_element - x2_element) for (x1_element, x2_element) in zip(x1, x2)])

def cosine_distance(x1, x2):
    cs_num = sum([x1_i * x2_i for (x1_i, x2_i) in zip(x1, x2)])
    cs_den = math.sqrt(sum([x1_i ** 2 for x1_i in x1])) * math.sqrt(sum([x2_i ** 2 for x2_i in x2]))
    cs = cs_num / cs_den
    cd = 1 - cs
    return cd