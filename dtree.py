# Implementation of decision tree algorithm
# Developed by: http://m-taghizadeh.ir
# Github: https://github.com/M-Taghizadeh/PyDTree

import math
from colorama import init, Fore
from csv import reader

# just for init colorama: (for cmd)
init()

# Load a CSV file
def load_csv(filename):
	file = open(filename, "rt")
	lines = reader(file)
	dataset = list(lines)
	return dataset


# type casting str to digit for csv raw data(is string by default)
def validation_on_numbers(dataset):
    for i in range(0, len(dataset)):
        for j in range(0, len(dataset[i])):
            if dataset[i][j].isdigit():
                dataset[i][j] = float(dataset[i][j])
    return dataset[1:] # split header


# Get dict contain that {class_name: number_of_samples_for_this_class}
def get_class_counts(rows):    
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts
# Test:
# print(get_class_counts(training_data)) # {'No': 7, 'Yes': 3}


# Test if a value is numeber or categorical value
def is_number(value):
    return isinstance(value, int) or isinstance(value, float)
# Test:
# is_number(7) # True
# is_number("Red") # Flase


# A Condition is used to partition a dataset.
# This class records a column_number:column_value
class Condition:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this condition.
        val = example[self.column]
        if is_number(val):
            return val >= self.value # for numerical attrs
        else:
            return val == self.value # for categorical attrs

    def __repr__(self):
        # This is just a helper method to print
        # the condition in a readable format.
        condition = "=="
        if is_number(self.value):
            condition = ">="
        return "%s %s %s?" % (
            header[self.column], condition, str(self.value))
# 1. Test condition for a numeric attribute
# print(Condition(1, 3))
# 2. Test condition for a numeric attribute
# c = Condition(0, 'Yes')
# checking training_data[0] matches with condition or not
# print(c, "=> result: ", c.match(training_data[0]))


# For each row in the dataset, check if it matches the condition. If so, add it to 'true rows', otherwise, add it to 'false rows'
def partition(rows, condition):
    true_rows, false_rows = [], []
    for row in rows:
        if condition.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    print("true_rows: ", true_rows, " false_rows: ", false_rows)
    return true_rows, false_rows


def entropy(rows):
    counts = get_class_counts(rows)
    e = 0
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        e += (-1)*(prob_of_lbl) * math.log2(prob_of_lbl)
    return e


# Information Gain = The entropy of the starting node - the weighted entropy of two child nodes.
def info_gain(left, right, current_entropy):
    print("Entropy(left): ", entropy(left), ", Entropy(right): ", entropy(right))
    p = float(len(left)) / (len(left) + len(right))
    return current_entropy - p * entropy(left) - (1 - p) * entropy(right)


# Find the best condition to ask by iterating over every feature/value and calculating the information gain
def find_best_split(rows):

    # Info Gain At least it must be greater than zero 
    best_gain = 0  
    best_condition = None 
    current_entropy = entropy(rows)
    n_features = len(rows[0]) - 1  # number of columns

    print("-----------------------------------------------------------------------------")
    print("Current entropy: ", current_entropy)

    for col in range(n_features):  # for each feature
        values = set([row[col] for row in rows])  # unique values in the column
        for val in values:  # for each value
            condition = Condition(col, val)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, condition)

            # Skip this split because that is a waste split :)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_entropy)
            print(Fore.BLUE, condition, Fore.RESET, " => Gain: ", gain, "\n")
            
            if gain >= best_gain:
                best_gain, best_condition = gain, condition

    print(Fore.GREEN, best_condition, Fore.RESET, " => Gain: ", best_gain)
    if best_gain<=0: 
        print(Fore.RED, "Dont Split!", Fore.RESET)

    return best_gain, best_condition
# Test:
# best_gain, best_condition = find_best_split(training_data)
# print(best_gain, best_condition)

class Leaf:
    def __init__(self, rows):
        self.predictions = get_class_counts(rows)


class Decision_Node:
    def __init__(self, condition, true_branch, false_branch):
        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch

# tree creator => Recursive function
def build_tree(rows):

    # calculate the information gain for each of the unique attributes and return the condition that produces the highest gain.
    gain, condition = find_best_split(rows)

    # Base case: 
    if gain == 0:
        return Leaf(rows)

    # Create Partitions or Two Way: 1.true_rows, 2.false_rows
    true_rows, false_rows = partition(rows, condition)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows)

    # return best node (best condition with best gain)
    return Decision_Node(condition, true_branch, false_branch)


def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Leaf", node.predictions)
        return

    # Print the condition at this node
    print (spacing + str(node.condition))
    # Call this function recursively on the true branch
    print (spacing + '└───True:')
    print_tree(node.true_branch, spacing + "\t")

    # Call this function recursively on the false branch
    print (spacing + '└───False:')
    print_tree(node.false_branch, spacing + "\t")


def classify(row, node):

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    if node.condition.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


if __name__ == '__main__':

    # 1. Read Training Set , The last column is the label and other columns are features.
    raw_csv_training_data = load_csv("dset.csv")
    header = raw_csv_training_data[0]
    training_data = validation_on_numbers(raw_csv_training_data)

    # 2. Create Decision Tree 
    my_tree = build_tree(training_data)
    print("-------------------------------------------------------")
    print(Fore.LIGHTYELLOW_EX, "\n[Decision Tree]\n")
    print_tree(my_tree)
    print(Fore.RESET)
    print("-------------------------------------------------------")

    # 3. Loading Testset:
    raw_csv_testing_data = load_csv("tset.csv")
    testing_data = validation_on_numbers(raw_csv_testing_data)

    # 4. Evaluate
    for row in testing_data:
        print("Sample: %s Label: %s => Predicted: %s" %(row[:-1], row[-1], print_leaf(classify(row, my_tree))))
