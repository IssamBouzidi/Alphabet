#coding:utf-8

import os
import random
import shutil
import numpy

# Creation of a sample of emails among all the emails.

def create_sample (number_of_sample, sample_size) :
    print(f"Creation of {number_of_sample} sample(s) each contening {sample_size} image(s).")

# Definition of the different directories.

    dirname = os.getcwd()
    imageDir = os.path.join(dirname,'data/RAW/alphabet-dataset/A')

# Counting emails.

    images_count = sum(len(files) for _, _, files in os.walk(imageDir)) 
    print(f"The imagedir folder contains {images_count} image(s).")

# Cleaning the target file.

    shutil.rmtree(os.path.join(dirname + os.sep + 'data' + os.sep + 'TRAINING' + os.sep + 'sample'))

    for sample in range(1, number_of_sample+1) :
        print (f"Sample number {sample}.")
        sampleDir = os.path.join(dirname,'data/TRAINING/sample', str(sample))

        if not os.path.exists(sampleDir):
            os.makedirs(sampleDir)

        # Draw random emails and copy them to the sample folder.
        
        random_list = numpy.random.randint(1,images_count+1,sample_size)
        print(len(random_list))
        id_image = 1
        for repertory, sub_repertory, files in os.walk(imageDir):
            for f in files :
                if id_image in random_list :
                    shutil.copy(os.path.join(repertory, f), sampleDir)
                    os.rename(os.path.join(sampleDir, f),os.path.join(sampleDir, str(id_image)))
                id_image +=1
        print (f"Creation of the sample {sample} successfully completed.")
        print (f"{sample_size} random image(s) have been copied to target repertory.")
    
    print (f"OK - {number_of_sample} sample(s) of {sample_size} image(s) created in target repertory.")

create_sample(1,10)

