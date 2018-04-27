#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/intropylab-classifying-images/check_images.py
#                                                                             
# TODO: 0. Fill in your information in the programming header below
# PROGRAMMER: Narath Chiev
# DATE CREATED: 04/24/2018
# REVISED DATE:             <=(Date Revised - if any)
# PURPOSE: Check images & report results: read them in, predict their
#          content (classifier), compare prediction to actual value labels
#          and output results
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
from time import time, sleep
from os import listdir

# Imports classifier function for using CNN to classify images 
from classifier import classifier 

# Imports argparse python module
import argparse

# Main program function defined below
def main():
    # TODO: 1. Define start_time to measure total program runtime by
    # collecting start time
    start_time = time()
    
    # TODO: 2. Define get_input_args() function to create & retrieve command
    # line arguments
    in_arg = get_input_args()
    
    # TODO: 3. Define get_pet_labels() function to create pet image labels by
    # creating a dictionary with key=filename and value=file label to be used
    # to check the accuracy of the classifier function
    answers_dic = get_pet_labels(in_arg.dir)

    # TODO: 4. Define classify_images() function to create the classifier 
    # labels with the classifier function uisng in_arg.arch, comparing the 
    # labels, and creating a dictionary of results (result_dic)
    result_dic = classify_images(in_arg.dir, answers_dic, in_arg.arch)

    # TODO: 5. Define adjust_results4_isadog() function to adjust the results
    # dictionary(result_dic) to determine if classifier correctly classified
    # images as 'a dog' or 'not a dog'. This demonstrates if the model can
    # correctly classify dog images as dogs (regardless of breed)
    adjust_results4_isadog(result_dic, in_arg.dogfile)

    # TODO: 6. Define calculates_results_stats() function to calculate
    # results of run and puts statistics in a results statistics
    # dictionary (results_stats_dic)
    results_stats_dic = calculates_results_stats(result_dic)

    # TODO: 7. Define print_results() function to print summary results, 
    # incorrect classifications of dogs and breeds if requested.
    print_results(result_dic, results_stats_dic, in_arg.arch)

    # TODO: 1. Define end_time to measure total program runtime
    # by collecting end time
    end_time = time()

    # TODO: 1. Define tot_time to computes overall runtime in
    # seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    
    # Prints overall runtime in seconds
    print("\nTotal Elapsed Runtime:", tot_time, "in seconds.")
           
    # Prints overall runtime in format hh:mm:ss
    print("\nTotal Elapsed Runtime:", str( int( (tot_time / 3600) ) ) + ":" + 
          str( int(  ( (tot_time % 3600) / 60 )  ) ) + ":" + 
          str( int( ( (tot_time % 3600) % 60 ) ) ) )

# TODO: 2.-to-7. Define all the function below. Notice that the input 
# paramaters and return values have been left in the function's docstrings. 
# This is to provide guidance for acheiving a solution similar to the 
# instructor provided solution. Feel free to ignore this guidance as long as 
# you are able to acheive the desired outcomes with this lab.

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object. 
     3 command line arguements are created:
       dir - Path to the pet image files(default- 'pet_images/')
       arch - CNN model architecture to use for image classification(default-
              pick any of the following vgg, alexnet, resnet)
       dogfile - Text file that contains all labels associated to dogs(default-
                'dognames.txt'
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()
    
	# Argument 1: that's a path to a folder
    parser.add_argument('--dir', type = str, default = 'pet_images/', 
						help = 'path to the folder pet_images') 

	# Argument 2: arch argument
    parser.add_argument('--arch', type = str, default = 'vgg', 
                        help = 'CNN model architecture to use for image classification')
    
    # Argument 3: that's an integer
    parser.add_argument('--dogfile', type = str, default = 'dognames.txt', 
                        help = 'The file that contains the list of valid dognames')

    return(parser.parse_args())

def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels based upon the filenames of the image 
    files. Reads in pet filenames and extracts the pet image labels from the 
    filenames and returns these label as petlabel_dic. This is used to check 
    the accuracy of the image classifier model.
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by pretrained CNN models (string)
    Returns:
     petlabels_dic - Dictionary storing image filename (as key) and Pet Image
                     Labels (as value)  
    """
    filename_list = listdir(image_dir)
    petlabels_dic = dict()
    
    for filename in filename_list:
      # lowers the name of the file
      pet_image_name = filename.lower()
      
      # splits file name by '_'
      word_list_image_name = pet_image_name.split('_')
      
      # pets name to be value of dictionary
      pet_name = ''
      
      # concats names, only if alphabetic
      # IE: ['small', 'dog', '04252018.jpg'] = 'small dog'  
      for word in word_list_image_name:
        if word.isalpha():
          pet_name += word + ' '
          
      # trim off any extra spaces
      pet_name = pet_name.strip()
      
      # updates or adds key value to dictionary 
      petlabels_dic[filename] = pet_name
      
    return(petlabels_dic)
  
def classify_images(images_dir, petlabel_dic, model):
    """
    Creates classifier labels with classifier function, compares labels, and 
    creates a dictionary containing both labels and comparison of them to be
    returned.
     PLEASE NOTE: This function uses the classifier() function defined in 
     classifier.py within this function. The proper use of this function is
     in test_classifier.py Please refer to this program prior to using the 
     classifier() function to classify images in this function. 
     Parameters: 
      images_dir - The (full) path to the folder of images that are to be
                   classified by pretrained CNN models (string)
      petlabel_dic - Dictionary that contains the pet image(true) labels
                     that classify what's in the image, where its' key is the
                     pet image filename & it's value is pet image label where
                     label is lowercase with space between each word in label 
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
     Returns:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)   where 1 = match between pet image and 
                    classifer labels and 0 = no match between labels
    """
    # dictionary to return classification
    results_dic = dict()
    
    # iterate through each pet label
    for filename, pet_name in petlabel_dic.items():
      # classifies the image with the filename 
      file_path_name = images_dir + filename
      image_classification = classifier(file_path_name, model)
      
      # lowercases the classification
      image_classification = image_classification.lower().strip()
      
      # checks whether pet name matches with the image classification results
      # if match then 1 else 0
      is_classification_matched = 1 if image_classification.find(pet_name) >= 0 else 0
      
      # updates the dictionary key= filename, value = [pet_name, image_classification, if pet_name matches classification]
      results_dic[filename] = [pet_name, image_classification, is_classification_matched]
    
    # dictionary to return back
    return (results_dic)

def adjust_results4_isadog(results_dic, dogsfile):
    """
    Adjusts the results dictionary to determine if classifier correctly 
    classified images 'as a dog' or 'not a dog' especially when not a match. 
    Demonstrates if model architecture correctly classifies dog images even if
    it gets dog breed wrong (not a match).
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    --- where idx 3 & idx 4 are added by this function ---
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
     dogsfile - A text file that contains names of all dogs from ImageNet 
                1000 labels (used by classifier model) and dog names from
                the pet image files. This file has one dog name per line
                dog names are all in lowercase with spaces separating the 
                distinct words of the dogname. This file should have been
                passed in as a command line argument. (string - indicates 
                text file's name)
    Returns:
           None - results_dic is mutable data type so no return needed.
    """
    # stored list of dog names
    dogs_names_list = []
    
    # append each dog name to the list, only strip since file should already be lowercase and each name is separated by line
    # also makes sure each item is unique in the list
    with open(dogsfile) as f:
      for line in f:
        if line not in dogs_names_list:
          dogs_names_list.append(line.rstrip())
    
    # iterate through each dictionary to is dog by petname, and is dog by classifier
    for filename, pet_classification in results_dic.items():
      # finds is-dog by petname
      pet_name_is_dog = 1 if pet_classification[0] in dogs_names_list else 0
      # finds is-dog by classification
      pet_classification_is_dog = 1 if pet_classification[1] in dogs_names_list else 0
      
      # appends the values to the results list value
      pet_classification.extend([pet_name_is_dog, pet_classification_is_dog])

def calculates_results_stats(results_dic):
    """
    Calculates statistics of the results of the run using classifier's model 
    architecture on classifying images. Then puts the results statistics in a 
    dictionary (results_stats) so that it's returned for printing as to help
    the user to determine the 'best' model for classifying images. Note that 
    the statistics calculated as the results are either percentages or counts.
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
    Returns:
     results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value 
    """
    
    # dictionary to return after calculation is complete
    results_stats = dict()
    # count of total images
    results_stats['n_img'] = len(results_dic)
    
    # counts to calculate after iterating through results
    results_stats['n_dogs_img'] = 0
    results_stats['n_correct_dogs'] = 0
    results_stats['n_correct_not_dogs'] = 0
    results_stats['n_correct_breed'] = 0
    results_stats['n_not_dogs'] = 0
    results_stats['n_label_matches'] = 0
    # iterate through list can count up the calculations
    for filename, result in results_dic.items():
      if result[3] == 1:
        results_stats['n_dogs_img'] += 1
        if result[4] == 1:
          results_stats['n_correct_dogs'] += 1
        if result[2] == 1:
          results_stats['n_correct_breed'] += 1
      if result[3] == 0 and result[4] == 0:
          results_stats['n_correct_not_dogs'] += 1
      if result[2] == 1:
          results_stats['n_label_matches'] += 1
    # Count of images that are not dogs
    results_stats['n_not_dogs'] =  results_stats['n_img'] - results_stats['n_dogs_img']
    # calulate percentages
    results_stats['pct_correct_dogs'] = (results_stats['n_correct_dogs'] / results_stats['n_dogs_img']) * 100
    results_stats['pct_correct_non_dogs'] = (results_stats['n_correct_not_dogs'] / results_stats['n_not_dogs']) * 100
    results_stats['pct_correct_breed'] = (results_stats['n_correct_breed'] / results_stats['n_dogs_img']) * 100
    results_stats['pct_label_matches'] = (results_stats['n_label_matches'] / results_stats['n_img']) * 100
	
    return(results_stats)

def print_results(results_dic, results_stats, model, print_incorrect_dogs=False, print_incorrect_breed=False):
    """
    Prints summary results on the classification and then prints incorrectly 
    classified dogs and incorrectly classified dog breeds if user indicates 
    they want those printouts (use non-default values)
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
      results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value 
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and 
                             False doesn't print anything(default) (bool)  
      print_incorrect_breed - True prints incorrectly classified dog breeds and 
                              False doesn't print anything(default) (bool) 
    Returns:
           None - simply printing results.
    """    
    print('Results for CNN model: {}'.format(model))
    print('\nNumber if Images: {} \nNumber of Dog Image: {} \nNumber of \"Not-a\" Dog Images {}'
          .format(
             results_stats['n_img']
            ,results_stats['n_dogs_img']
            ,results_stats['n_not_dogs']
          )
         )
    print('\n% Not-a-Dog-Correct {} \n% Dogs Correct: {} \n% Breeds Correct {} \n% Labels Match {}'
          .format(
             results_stats['pct_correct_non_dogs']
            ,results_stats['pct_correct_dogs']
            ,results_stats['pct_correct_breed']
            ,results_stats['pct_label_matches']
          )
         )

# Call to main function to run the program
if __name__ == "__main__":
    main()
