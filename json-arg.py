# importing all the necessary libraries
import sys
import argparse
import csv
import json
import cv2
import torch 
import git
import pathlib
import pandas as pd
from pathlib import Path
from collections import defaultdict
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from deep_sort_realtime.deepsort_tracker import DeepSort
from moviepy.editor import VideoFileClip, concatenate_videoclips


# parser to accept cmd line argumnets
parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('output', type=str)
args = parser.parse_args()

# storing path data from parser
input_json_path = args.input
output_json_path = args.output


try:
  # Check if running on Windows (drive letter)
  if str(pathlib.Path.cwd())[:1].upper() == 'C':
    pathlib.PosixPath = pathlib.WindowsPath  # Override PosixPath with WindowsPath
except AttributeError:
  # If 'PosixPath' attribute doesn't exist (likely non-Windows system)
  pass


# Cloning the Yolov5 Repo
repo_path1 = Path('yolov5m')
if not repo_path1.exists():
    print(f'Cloning YOLOv5 repository to {repo_path1}...')
    git.Repo.clone_from('https://github.com/ultralytics/yolov5', repo_path1)
else:
    print(f'Repository {repo_path1} already exists.')


repo_path1=Path('yolov5m')
sys.path.append(str(repo_path1))

# function to read and load the contents of the json file
def read_json():
    with open(input_json_path, 'r') as file:
        data = json.load(file)
    return data

# function to extract camid and videopaths from json file
def extract_data(json_data):
    key = list(json_data.keys())[0] 
    cam_id = json_data[key]  
    video1 = cam_id.get('Vid_1')  
    video2 = cam_id.get('Vid_2') 
    return key,video1,video2


# retreiving all the required info from json file
json_data = read_json()
cam_id , video1_path , video2_path = extract_data(json_data)


def region_Stn_HD_1(frame):
    h, w, _ = frame.shape
    regions = {
        'A': (0, h//2-85, w//6+300, h),                 # Bottom-left
        'B': (0, 0, w//6+100, h//2),                    # Top-left
        'C': (w//6, 0, 5*w//6-270, h//2),               # Top-center
        'D': (5*w//6-270, 0, w-180, h//2),              # Top-right
        'E': ( w-180, 0, w, h//2),                      # Right
        'F': (w//6+200, h//2-82, 5*w//6+130, h)     # Bottom-center
    }
    return regions

def transition_Stn_HD_1():
    return {'BC', 'BE', 'DE', 'DA', 'FA', 'FC'}


def region_Sty_Wll_Ldge_FIX_3(frame):
    h, w, _ = frame.shape
    regions = {
            'A': (0, 0, w//4-30, h-50),   # objects detected in A make the transition of BA
            'B': (w//4-25,0, w, h)      # objects detected in B make the transition of AB   
    }
    return regions

def transition_Sty_Wll_Ldge_FIX_3():
    return {'AB', 'BA'}


def region_SBI_Bnk_JN_FIX_1(frame):
    h, w, _ = frame.shape
    regions = {
        'A': (0, h//2-140, w//6+900, h),                 # Bottom-left
        'B': (0, 0, w//6+900, h//2-150),                    # Top-left
        'C': (w//6+905, 0, w, h)
    } 

    return 0

def transition_SBI_Bnk_JN_FIX_1():
    return { 'AB', 'AC', 'BA', 'BC', 'CA', 'CB'}


def region_SBI_Bnk_JN_FIX_3(frame):
    h, w, _ = frame.shape
    regions = {
        'A': (0, 0, w, h//2),                 # Bottom-left
        'B': (0, h//2+5, w, h)
    }    

    return 0

def transition_SBI_Bnk_JN_FIX_3():
    return {'AB','BA'}


def region_18th_Crs_BsStp_JN_FIX_2(frame):
    h, w, _ = frame.shape
    regions = {
        'A': (0, 0, w, h//2),                 # Bottom-left
        'B': (0, h//2+5, w, h)
    }

    return regions

def transition_18th_Crs_BsStp_JN_FIX_2():
    return {'AB','BA'}


def region_18th_Crs_Bus_Stop_FIX_2(frame):
    h, w, _ = frame.shape
    regions = {
        'A': (0, 0, w//4-70, h),                 # Bottom-left
        'B': (w//4-65, 0, w//4+300, h//4), 
        'C': (w//4+305, 0, w//4+710, h//4),
        'D': (w//4+712, 0, w, h-570),
        'E': (w//4-69, h//2+100, w, h)
    }
    return regions

def transition_18th_Crs_Bus_Stop_FIX_2():
    return {' AB', 'AD', 'CD', 'EB', 'ED'}


def region_Ayyappa_Temple_FIX_1(frame):
    h, w, _ = frame.shape
    regions = {
        'A': (0, h//2+5, w, h),                 # Bottom-left
        'B': (0, 0, w, h//2)
    }

    return regions
 
def transition_Ayyappa_Temple_FIX_1():
    return {'AB','BA'}


def region_Devasandra_Sgnl_JN_FIX_1(frame):
    h, w, _ = frame.shape
    regions = {
        'A': (0, h//2-90, w, h),                
        'B': (0, 0, w, h//2-100)
    }
    return regions

def transition_Devasandra_Sgnl_JN_FIX_1():
    return {'AB','BA'}


def region_Devasandra_Sgnl_JN_FIX_3(frame):
    h, w, _ = frame.shape
    regions = {
        'A': (0, h//4-100, w//2-200, h),               
        'B': (0, 0, w//8+270, h//8), 
        'C': (w//4+300, 0, w-300, h//4-70),
        'D': (w//2+212, h//4-70+3, w, h)        
    }
    return regions

def transition_Devasandra_Sgnl_JN_FIX_3():
    return { 'AB', 'AD', 'CA', 'CD', 'DA', 'DB'}


def region_Mattikere_JN_FIX_1(frame):
    h, w, _ = frame.shape
    regions = {
        'A': (0, h//6-100, w//6+150, 3 * h // 2),               
        'B': (w // 4, h//6-100, 2*w//3-120, 3*h//7), 
        'C': (2*w//4+210, h//4-200, w, 3*h//4-200),
        'D': (2 * w // 3-400,3 * h // 4 - 190, w, h)        
    }
    return regions

def transition_Mattikere_JN_FIX_1():
    return { 'AB', 'AD', 'CA', 'CD', 'DA', 'DB'}


def region_Mattikere_JN_FIX_2(frame):
    h, w, _ = frame.shape
    regions = {
        'A': (w // 2 - 700, 0,w // 4 + 200, h // 3),              
        'B': (w // 2 - 275, 0,2*w//4 + 290, 3 * h // 7 - 100), 
        'C': (2*w // 4+410, h // 4 - 200, w, 3 * h // 4-300),
        'D': ( 0, 3 * h // 4 - 290, w, h)       
    }
    return regions

def transition_Mattikere_JN_FIX_2():
    return { 'BC', 'BD', 'CA', 'CD', 'DA', 'DC'}


def region_Mattikere_JN_FIX_3(frame):
    h, w, _ = frame.shape
    regions = {
        'A': (0, 0,w//3 - 100, h ),              
        'B': (w // 2 - 400 , 0, w  + 200, h) 
    }
    return regions

    return 0

def transition_Mattikere_JN_FIX_3():
    return {'AB','BA'}


def region_Mattikere_JN_HD_1(frame):
    h, w, _ = frame.shape
    regions = {
        'A': (0, h // 3 - 400,w // 4 + 200, h // 2 - 50),              
        'B': (w // 3 + 50, h // 5 - 500,3 * w // 4 + 400, h // 3 - 20), 
        'C': (3 * w // 4, h // 3,w - 20 + 20, 2 * h // 3 + 500),
        'D': (w // 28 - 200, h // 2 - 45,w // 2 +450, h - 20)
    }
    return regions

def transition_Mattikere_JN_HD_1():
    return {' AD', 'AC', 'BA', 'BD', 'CA', 'CD'}    


def region_HP_Ptrl_Bnk_BEL_Rd_FIX_2(frame):
    h, w, _ = frame.shape
    regions = {
        'A':  (0, 0, w, h // 2 - 50),              
        'B':  (0, h // 2 - 40, w , h) 
    }
    return regions    

    return 0

def transition_HP_Ptrl_Bnk_BEL_Rd_FIX_2():
    return {'AB','BA'} 


def region_Kuvempu_Circle_FIX_1(frame):
    h , w , _ = frame.shape
    regions = {
        'A': (0, 0, w//2-200, h),        
        'B': (w//2-200, 0, w, h)    
     }
    return regions

def transition_Kuvempu_Circle_FIX_1():
    return {'AB','BA'}     

    
def region_Kuvempu_Circle_FIX_2(frame):
    h , w , _ = frame.shape
    regions = {
        'A': (0, 0, w//2-200, h),        
        'B': (w//2-200, 0, w, h)      
    }
    return regions

def transition_Kuvempu_Circle_FIX_2():
    return {'AB','BA'}   


def region_MS_Ramaiah_JN_FIX_1(frame):
    h, w, _ = frame.shape
    regions = {
         'A': (w//5, 0, 3*w//5, 2*h//4),      
          'B': (2*w//3, 0, w, h),    
          'C': (0, 2*h//5, w//2, h)        
    }
    return regions

def transition_MS_Ramaiah_JN_FIX_1():
    return {'AB','AC'}  


def region_MS_Ramaiah_JN_FIX_2(frame):
    h, w, _ = frame.shape
    regions = {
        'A': (0, h//2 - 100,w//4-100, h),                 # Bottom-left
        'B': (0, 0,w //4, h // 3 + 70), 
        'C': (w // 4 + 50, 0,3 * w // 7 - 20, h // 3 + 70),
        'D': (2*w // 5 + 50, 0,2 * w // 3 - 20, h // 3 + 70),
        'E': (2 * w // 3, 0,w, 4*h // 9 + 140),
        'F': (2*w//3, h//1 + 200,w, 2*h//3 - 90),
        'G': (w//3+100, 2*h//3 - 60,2*w//2 - 650, h),
        'H': (w//6+30, 2*h//3 -50, w//3-200, h)
    }
    return regions
    

def transition_MS_Ramaiah_JN_FIX_2():
    return {'BC', 'BE', 'BG', 'DA', 'DE', 'DG', 'FA', 'FC', 'FG', 'HA', 'HC', 'HE'}   


def region_Ramaiah_BsStp_JN_FIX_1(frame):
    h, w, _ = frame.shape
    regions = {
         'A': (0, h//4, w, h),       
         'B': (0, 0, w //1, h // 3)
    }
    return regions    

def transition_Ramaiah_BsStp_JN_FIX_1():
    return {'AB','BA'}   


def region_Ramaiah_BsStp_JN_FIX_2(frame):
    h, w, _ = frame.shape
    regions = {
        'A': (0,  2*h // 5, w, h),  
        'B': (0, 0, w, 2* h // 5) 
    }
    return regions

def transition_Ramaiah_BsStp_JN_FIX_2():
    return {'AB','BA'}  


def point_in_region(point, region):
    x, y = point
    x1, y1, x2, y2 = region
    return x1 <= x <= x2 and y1 <= y <= y2

def get_region(x, y, regions):
    for region_name, region_coords in regions.items():
        if point_in_region((x, y), region_coords):
            return region_name
    return None


# mapping different camid to their relevant function names
case_statement0 = {
    'Stn_HD_1': region_Stn_HD_1,
    'Sty_Wll_Ldge_FIX_3': region_Sty_Wll_Ldge_FIX_3,
    'SBI_Bnk_JN_FIX_1':region_SBI_Bnk_JN_FIX_1,
    'SBI_Bnk_JN_FIX_3':region_SBI_Bnk_JN_FIX_3,
    '18th_Crs_BsStp_JN_FIX_2':region_18th_Crs_BsStp_JN_FIX_2,
    '18th_Crs_Bus_Stop_FIX_2':region_18th_Crs_Bus_Stop_FIX_2,    
    'Ayyappa_Temple_FIX_1':region_Ayyappa_Temple_FIX_1,
    'Devasandra_Sgnl_JN_FIX_1':region_Devasandra_Sgnl_JN_FIX_1,
    'Devasandra_Sgnl_JN_FIX_3':region_Devasandra_Sgnl_JN_FIX_3,    
    'Mattikere_JN_FIX_1':region_Mattikere_JN_FIX_1,
    'Mattikere_JN_FIX_2':region_Mattikere_JN_FIX_2,
    'Mattikere_JN_FIX_3':region_Mattikere_JN_FIX_3,    
    'Mattikere_JN_HD_1':region_Mattikere_JN_HD_1,
    'HP_Ptrl_Bnk_BEL_Rd_FIX_2':region_HP_Ptrl_Bnk_BEL_Rd_FIX_2,
    'Kuvempu_Circle_FIX_1':region_Kuvempu_Circle_FIX_1,    
    'Kuvempu_Circle_FIX_2':region_Kuvempu_Circle_FIX_2,
    'MS_Ramaiah_JN_FIX_1':region_MS_Ramaiah_JN_FIX_1,
    'MS_Ramaiah_JN_FIX_2':region_MS_Ramaiah_JN_FIX_2,
    'Ramaiah_BsStp_JN_FIX_1':region_Ramaiah_BsStp_JN_FIX_1,
    'Ramaiah_BsStp_JN_FIX_2':region_Ramaiah_BsStp_JN_FIX_2
}

case_statement1 = {
    'Stn_HD_1': transition_Stn_HD_1,
    'Sty_Wll_Ldge_FIX_3': transition_Sty_Wll_Ldge_FIX_3,
    'SBI_Bnk_JN_FIX_1':transition_SBI_Bnk_JN_FIX_1,
    'SBI_Bnk_JN_FIX_3':transition_SBI_Bnk_JN_FIX_3,
    '18th_Crs_BsStp_JN_FIX_2':transition_18th_Crs_BsStp_JN_FIX_2,
    '18th_Crs_Bus_Stop_FIX_2':transition_18th_Crs_Bus_Stop_FIX_2,    
    'Ayyappa_Temple_FIX_1':transition_Ayyappa_Temple_FIX_1,
    'Devasandra_Sgnl_JN_FIX_1':transition_Devasandra_Sgnl_JN_FIX_1,
    'Devasandra_Sgnl_JN_FIX_3':transition_Devasandra_Sgnl_JN_FIX_3,    
    'Mattikere_JN_FIX_1':transition_Mattikere_JN_FIX_1,
    'Mattikere_JN_FIX_2':transition_Mattikere_JN_FIX_2,
    'Mattikere_JN_FIX_3':transition_Mattikere_JN_FIX_3,    
    'Mattikere_JN_HD_1':transition_Mattikere_JN_HD_1,
    'HP_Ptrl_Bnk_BEL_Rd_FIX_2':transition_HP_Ptrl_Bnk_BEL_Rd_FIX_2,
    'Kuvempu_Circle_FIX_1':transition_Kuvempu_Circle_FIX_1,    
    'Kuvempu_Circle_FIX_2':transition_Kuvempu_Circle_FIX_2,
    'MS_Ramaiah_JN_FIX_1':transition_MS_Ramaiah_JN_FIX_1,
    'MS_Ramaiah_JN_FIX_2':transition_MS_Ramaiah_JN_FIX_2,
    'Ramaiah_BsStp_JN_FIX_1':transition_Ramaiah_BsStp_JN_FIX_1,
    'Ramaiah_BsStp_JN_FIX_2':transition_Ramaiah_BsStp_JN_FIX_2
}


# combining two videos
clip1 = VideoFileClip(video1_path)
clip2 = VideoFileClip(video2_path)

final_clip = concatenate_videoclips([clip1, clip2])

final_clip.write_videofile("combined_video.mp4")


# checking device compatability
device='cuda'
if device == 'cuda' and not torch.cuda.is_available():
    print("CUDA is not available. Falling back to CPU.")
    device = 'cpu'
else:
    print("Cuda available")


# loading the yolo model 
model=torch.hub.load('ultralytics/yolov5', 'custom', path='mbest.pt', force_reload=True).to(device)


# Initialize object counter
transition_counts0 = defaultdict(lambda: defaultdict(int))
transition_counts1 = defaultdict(lambda: defaultdict(int))

# initialize the tracker
tracker = DeepSort(max_age=50, n_init=3, nn_budget=100)


# start capturing of the video for processing 
cap=cv2.VideoCapture('combined_video.mp4')


# Frame rate and frame count
frame_rate = cap.get(5)
int_frame=int(frame_rate)
frames_per_set = int_frame * 60  # 1 minutes in frames

# Initialize variables
frame_count = 0
set_count = 0
data = []

# Track the last known region of each object
object_regions = {}

# starting the main loop
while cap.isOpened():
    ret,frame=cap.read()

    if not ret:
        break

    frame_count += 1

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = model(frame_rgb)


    # function call to retrieve the regions
    if cam_id in case_statement0:
       regions=case_statement0[cam_id](frame)
    else:
       print('No matching cam_id found')


     # Prepare detections for Deep SORT
    detections = []
    detection_classes = {}

    for result in result.xyxy[0]:  # results.xyxy is a tensor
        x1, y1, x2, y2, conf, cls = result
        cls = int(cls)
        if cls in range(len(model.names)):
            width = (x2 - x1).item()
            height = (y2 - y1).item()
            left = x1.item()
            top = y1.item()
            confidence = conf.item()    
        
        if conf > 0.1 :
            # Append detection as a tuple in the format ([left, top, width, height], confidence, detection_class)
            detections.append(([left, top, width, height], confidence, cls))
            


    # Update tracker with RGB frame
    tracks = tracker.update_tracks(detections, frame=frame_rgb)



    # Process tracks
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_tlbr()
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2


        cls = int(cls)  # Convert to integer
        if cls in model.names:
            class_name = model.names[cls]

            current_region = get_region(x_center, y_center, regions)
            if track_id in object_regions:
              previous_region = object_regions[track_id]
              if previous_region and current_region:
                    transition_pattern = previous_region + current_region
                    if cam_id in case_statement1:
                        if transition_pattern in case_statement1[cam_id]():
                            transition_counts0[transition_pattern][class_name] += 1
                            transition_counts1[transition_pattern][class_name] += 1
                    
            object_regions[track_id] = current_region


     # Save data to CSV every 2 minutes
    if frame_count % frames_per_set == 0:
        # Flatten transition counts into a single dictionary
        flattened_counts = {}
        for pattern, class_dict in transition_counts1.items():
            for class_name, count in class_dict.items():
                flattened_counts[f'{class_name}_{pattern}'] = count

        data.append(flattened_counts)
        set_count += 1
        transition_counts1.clear()

  
cap.release()


# Convert the data into a DataFrame and save to CSV
df = pd.DataFrame(data)
df.fillna(0, inplace=True)

df.to_csv('data_for_pred.csv', index=False)

def make_stationary(series):
    result = adfuller(series)
    if result[1] > 0.05:  # p-value > 0.05, non-stationary
        series = series.diff().dropna()  # Differencing to make stationary
    return series


df['timestamp'] = pd.date_range(start='00:01:00', periods=len(df), freq='1min')
df.set_index('timestamp', inplace=True)


predictions = pd.DataFrame()
future_timestamps = pd.date_range(start=df.index[-1] + pd.Timedelta(minutes=1), periods=30, freq='1min')
predictions['timestamp'] = future_timestamps

if cam_id in case_statement1:
        turning_patterns = list(case_statement1[cam_id]())
vehicle_types = ['Car', 'Bus', 'Truck', 'Three-Wheeler', 'Two-Wheeler', 'LCV', 'Bicycle']


nested_predictions = {tp: {vt: [] for vt in vehicle_types} for tp in turning_patterns}

columns_to_predict = [f"{vt}_{tp}" for tp in turning_patterns for vt in vehicle_types]


predicted_counts = {tp: {vt: [] for vt in vehicle_types} for tp in turning_patterns}



for column in columns_to_predict:
    if column in df.columns:
        try:
            series = df[column]
            stationary_series = make_stationary(series)
            model = auto_arima(stationary_series, start_p=1, start_q=1,
                               max_p=3, max_q=3, seasonal=False,
                               trace=True, error_action='ignore', suppress_warnings=True)
            forecast = model.predict(n_periods=30)


            tp, vt = column.split('_')
            predicted_counts[tp][vt] = forecast.tolist()
        except Exception as e:
            print("Some column combinations are not available")

print(predicted_counts)


def save_transition_and_predictions_to_json(transition_counts0, predicted_counts, output_json_path, cam_id):
    # Define the vehicle types and turning patterns
    if cam_id in case_statement1:
        turning_patterns = list(case_statement1[cam_id]())
    vehicle_types = ['Car', 'Bus', 'Truck', 'Three-Wheeler', 'Two-Wheeler', 'LCV', 'Bicycle']

    # Initialize dictionaries for cumulative and predicted counts
    cumulative_counts = {tp: {vt: 0 for vt in vehicle_types} for tp in turning_patterns}
    predicted_counts_dict = {tp: {vt: 0 for vt in vehicle_types} for tp in turning_patterns}

    # Populate cumulative counts from transition_counts0
    for pattern in turning_patterns:
        if pattern in transition_counts0:
            for class_name in vehicle_types:
                cumulative_counts[pattern][class_name] = transition_counts0.get(pattern, {}).get(class_name, 0)

    # Populate predicted counts from predicted_counts
    for pattern in turning_patterns:
        if pattern in predicted_counts:
            for class_name in vehicle_types:
                predicted_counts_dict[pattern][class_name] = predicted_counts.get(pattern, {}).get(class_name, 0)

    # Prepare the final output JSON structure
    output_data = {
        cam_id: {
            "Cumulative Counts": cumulative_counts,
            "Predicted Counts": predicted_counts_dict
        }
    }

    # Save the JSON data to a file
    with open(output_json_path, 'w') as json_file:
        json.dump(output_data, json_file, indent=4)

    print(f'JSON data saved to {output_json_path}')




save_transition_and_predictions_to_json(transition_counts0, predicted_counts, output_json_path, cam_id)














