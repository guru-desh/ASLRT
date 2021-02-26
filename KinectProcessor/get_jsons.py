# get_jsons.py
# Converts Azure Kinect videos to Kinect feature json files. Requires python3, libk4a1.3-dev, and libk4abt0.9
# Example for videos with right-handed signers: python get_jsons.py /path/to/inputdirectory /path/to/outputdirectory right
# Example for videos with left-handed signers: python get_jsons.py /path/to/inputdirectory /path/to/outputdirectory left
import glob
import subprocess
import tqdm
import os

all_errors = "\nERRORS:\n"

def runOfflineProcessor(source: str, dest: str, is_left: bool):
    
    global all_errors

    if " " in source:
        source = f"\"{source}\""
    if " " in dest:
        dest = f"\"{dest}\""
    if is_left:
        command = f"./offline_processor_left {source} {dest}.json"
    else:
        command = f"./offline_processor {source} {dest}.json"
    try:
        output = subprocess.check_output(command, shell=True)
    except:
        print(f'{command} errored. please check {source}')
        all_errors += f'{command} errored. please check {source}\n'
    

def produceJSONS(source: str, destination: str, is_left: bool):

    global all_errors

    video_list = glob.glob(source, recursive=True)

    # print(video_list)

    for video in tqdm.tqdm(video_list):
        name = video.split("/")[-1].replace(".mkv", "")
        prefix = '/'.join(video.split("/")[-4:-1])
        destionationDir = os.path.join(destination, prefix)
        if not os.path.exists(destionationDir):
            os.makedirs(destionationDir)
            destinationFile = os.path.join(destination, prefix, name)
            # print(video, destinationFile, sep = ' | ')
            runOfflineProcessor(video, destinationFile, is_left)

    session = source.split('/')[-3]
    print(session)
    print(all_errors)
    f = open(f'Errors_{session}.txt', 'w')
    f.write(all_errors)
    f.close()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Format: python get_jsons.py (input directory, no trailing slash) (output directory, no trailing slash) (left/right, representing dominant hand of signer)")
    produceJSONS(sys.argv[0]+"/**/*.mkv", sys.argv[1]+"/", sys.argv[2] == "left")


