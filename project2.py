#Project 2, CS170 Winter 2026.
#Isabela Sforcin 
#For this project, I need to first develop a nearest neighbor algorithm, then I will implement a wrapper that uses the algorithm to perform forward and backward search

#I will first develop the nearest neighbor, then test it using datasets provided by the professor. Then I will implement the search algorithnms.
#For this project, I am attempting a better note-taking approach, as I have struggled with writing and reading comments that are nested in the code.

import time 
import numpy as np
import math




#The first step is to load the data from the txt file
def load_and_split_data(filename): 
    data = np.loadtxt(filename)                                     #the loadtxt function is included in numpy. It loads the data and stores it in an array. [1]
    labels = data[:, 0]
    features = data[:, 1:]
    return np.column_stack((labels, features))

#in order to do nearest neighbor, I will use one data point and compare it to the other rows in the dataset using distance
#Then, I will check if the nearest neighbor has the same label. 
#after that, I will calculate the amount of correct classifications (accuracy)
             #divide by dataset size to get accuracy.


#I updated this function using the numpy sum function, which enabled me to calculate all the differences at once
#and then taking the sqrt to get the distance

def nearest_neighbor_func(data, curr_features):
    number_correct = 0
    n = len(data)                                                    #Adding this made accessing the data faster 

    for i in range(n):
        object_i = data[i, curr_features]
        label_i= data[i,0]

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_label = None

        data_diff= object_i - data[:, curr_features]
        data_squared_diff = np.sum(data_diff**2, axis=1)             #using numpy sum [4], we add the distance from all data points                        
        for j in range(n):
            if i == j:                                               #We take one data point and compare to all other ones
                continue
            distance = math.sqrt(data_squared_diff[j])               #take the square root of the sum of squares to calculate the Euclidean distance

            if distance < nearest_neighbor_distance:                 #if distance is less, update the nearest neighbor
                nearest_neighbor_distance = distance
                nearest_neighbor_label = data[j,0]

        if label_i == nearest_neighbor_label:                       #then check if the label is the same
            number_correct += 1                                     #if so, it is correct, add the number correct 

    return number_correct / n


#Very important: Make sure your nearest neighbor algorithm is working correctly before you attempt the search algorithms. We will
#provide some test datasets for this purpose. [3]
#sanity check with provided dataset.
data = load_and_split_data("SanityCheck1.txt")                      #I changed the name of the file prof. Keogh gave us because it was easier to copy/paste
#from "Project 2 ReadME," by prof. Keogh on Dropbox.
#I utilized the sanity check to ensure the nearest neighbor classifier was working correctly, as suggested by professor Keogh.
test_features = [7, 10, 12]
print(f"Running nearest neighbor on Sanity Check 1 with features {test_features}...")
accuracy = nearest_neighbor_func(data, test_features)

# Check if it is within the 2% margin
print(f"Accuracy: {accuracy:.3f}")
#the accuracy was 0.950, which is within the 2% margin. Now I can move on to do the search algorithms.


#forward selection:
# forward selection starts with no features, and then adds one at a time (the most significant ones)
def forward_selection(data):
    num_features= len(data[0]) - 1
    curr_set =[]
    best_set=[]
    best_accuracy=0
    for level in range(1,num_features +1):                                          #go through all the levels
        feature_add=0
        best_acc=0
        for feature in range(1, num_features+1):                                    #if the feature is not in the set, add it and test the accuracy
            if feature not in curr_set:
                test_set = curr_set + [feature]
                accuracy = nearest_neighbor_func(data, test_set)                    #use the nearest neighbor to calculate the accuracy
                print("using feature",test_set, "the accuracy is", accuracy)
                if accuracy >best_acc:                                              # if the accuracy is better, add feature to the list and update best accuracy.
                    best_acc = accuracy
                    feature_add = feature
        curr_set.append(feature_add)
        print(f"On level {level}, added feature {feature_add} to current set, which has accuracy {best_acc}") 
        print()
        if best_acc > best_accuracy:                                                # if the subset accuracy is better, update the overall best accuracy
            best_accuracy = best_acc
            best_set = curr_set.copy()   
    print(f"Best set is {best_set} with accuracy {best_accuracy}")


#backward selection is the same type, but we start with all the features and remove the "bad" ones

def backward_selection(data):
    num_features= len(data[0]) - 1                                                  #start with all features
    curr_set =[]

    for feature in range(1, num_features+1):                                        #add all features to the current set
        curr_set.append(feature)
    best_set= curr_set.copy()
    best_accuracy= nearest_neighbor_func(data, curr_set)                            #calculate the accuracy of the full set, and then we will remove features one by one and check if the accuracy improves

    for level in range(1,num_features):
        feature_remove=0
        best_acc=0
        for feature in curr_set:
            test_set = curr_set.copy()
            test_set.remove(feature)
            accuracy = nearest_neighbor_func(data, test_set)                        #same logic as forward, but it checks if removing the feature improves the accuracy.
            print("using feature",test_set, "the accuracy is", accuracy)           #removed this for the 
            if accuracy >best_acc:
                best_acc = accuracy
                feature_remove = feature
        curr_set.remove(feature_remove)
        print(f"On level {level}, removed feature {feature_remove} from current set, which has accuracy {best_acc}")
        print()

        if best_acc > best_accuracy:
            best_accuracy = best_acc
            best_set = curr_set.copy()
    print(f"Best set is {best_set} with accuracy {best_accuracy}")


#now, I will implement the trace as it is shown on "Project 2 Winter 2026"
print ("Welcome to Isabela Sforcin's feature selection algorithm, type in the name of the file to test")
filename = input()
data = load_and_split_data(filename)
print("type (1) if you'd like forward selection, or (2) if you'd like backward selection")
selection_type = input()
if selection_type == "1":
    print("you have selected forward selection")
    start_time = time.time()
    forward_selection(data)
    end_time = time.time()
    print("Execution time:", end_time - start_time, "seconds")
elif selection_type == "2":
    start_time = time.time()
    print("you have selected backward selection")
    backward_selection(data)
    end_time = time.time()
    print("Execution time:", end_time - start_time, "seconds")








#citations:
# [1] "numpy.loadtxt — NumPy v2.0 Manual," NumPy, 2024. [Online]. Available: https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html. 
#[2] I took this logic from CS170 class, where prof. Keogh explained how Nearest Neighbor works, and how we update the NN as we see new data.
#[3] "Project 2 Winter 2026" by prof. Eamon Keogh on Dropbox.
#[4] Numpy sum Documentation https://numpy.org/doc/stable/reference/generated/numpy.sum.html
