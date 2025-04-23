import cv2
import time
import numpy as np
import pandas as pd
import redis
from datetime import datetime
import insightface
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys
from dotenv import load_dotenv

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

# Load environment variables
env_path = resource_path('.env') if getattr(sys, 'frozen', False) else '.env'
load_dotenv(env_path)

class FaceRecognitionSystem:
    def __init__(self):
        # Redis configuration from .env
        redis_config = {
            'host': os.getenv('REDIS_HOST', 'localhost'),
            'port': int(os.getenv('REDIS_PORT', 6379)),
            'password': os.getenv('REDIS_PASSWORD', None)
        }
        
        # Application settings from .env
        self.action_interval = int(os.getenv('ACTION_INTERVAL', 10))
        self.detection_threshold = float(os.getenv('DETECTION_THRESHOLD', 0.5))
        self.default_zone = os.getenv('DEFAULT_ZONE', 'Lagos Zone 2')
        
        # Initialize Redis connection
        self.r = redis.StrictRedis(**redis_config)
        
        # Initialize face analysis model with correct paths
        model_path = resource_path('insightface_model') if getattr(sys, 'frozen', False) else 'insightface_model'
        self.faceapp = FaceAnalysis(
            name='buffalo_sc',
            root=model_path,
            providers=['CPUExecutionProvider']
        )
        self.faceapp.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
        
        # Load staff database
        self.staff_db = self.retrieve_data('staff:register')
        self.current_action = None
        self.last_action_time = time.time()
        self.status_message = ""
        self.status_message_time = 0
        
    def retrieve_data(self, name):
        """Retrieve facial recognition data from Redis database"""
        retrive_dict = self.r.hgetall(name)
        retrive_series = pd.Series(retrive_dict)
        retrive_series = retrive_series.apply(lambda x: np.frombuffer(x, dtype=np.float32))
        index = retrive_series.index
        index = list(map(lambda x: x.decode(), index))
        retrive_series.index = index
        retrive_df = retrive_series.to_frame().reset_index()
        retrive_df.columns = ['ID_Name_Role', 'Facial_features']
        
        retrive_df['File No. Name'] = ''
        retrive_df['Role'] = ''
        retrive_df['Zone'] = self.default_zone
        
        for i, row in retrive_df.iterrows():
            try:
                parts = row['ID_Name_Role'].split('@')
                file_name_role = parts[0]
                file_no, name = file_name_role.split('.', 1)
                role = parts[1] if len(parts) > 1 else ''
                zone = parts[2] if len(parts) > 2 else self.default_zone
                
                retrive_df.at[i, 'File No. Name'] = f"{file_no}.{name}"
                retrive_df.at[i, 'Role'] = role
                retrive_df.at[i, 'Zone'] = zone
                
            except Exception as e:
                print(f"Error processing record {row['ID_Name_Role']}: {str(e)}")
                continue
        
        return retrive_df[['ID_Name_Role', 'File No. Name', 'Role', 'Facial_features', 'Zone']]
    
    def ml_search_algorithm(self, test_vector, thresh=0.5):
        """Perform facial recognition search using cosine similarity"""
        dataframe = self.staff_db.copy()
        X_list = dataframe['Facial_features'].tolist()
        X_cleaned = []
        indices = []

        for i, item in enumerate(X_list):
            if isinstance(item, (list, np.ndarray)) and len(item) > 0:
                item_arr = np.array(item)
                if item_arr.shape == test_vector.shape:
                    X_cleaned.append(item_arr)
                    indices.append(i)

        if len(X_cleaned) == 0:
            return 'Unknown', 'Unknown'

        dataframe = dataframe.iloc[indices].reset_index(drop=True)
        x = np.array(X_cleaned)
        similar = cosine_similarity(x, test_vector.reshape(1, -1))
        similar_arr = similar.flatten()
        dataframe['cosine'] = similar_arr
        data_filter = dataframe[dataframe['cosine'] >= thresh]

        if not data_filter.empty:
            best_match = data_filter.sort_values(by='cosine', ascending=False).iloc[0]
            person_name, person_role = best_match['File No. Name'], best_match['Role']
        else:
            person_name, person_role = 'Unknown', 'Unknown'

        return person_name, person_role
    
    def check_last_action(self, name, current_action):
        """Check if the current action is valid based on previous logs"""
        if name == 'Unknown':
            return True
        
        logs = self.r.lrange('attendance:logs', 0, 10)
        last_action = None
        last_date = None
        
        for log in logs:
            if isinstance(log, bytes):
                log = log.decode('utf-8')
            
            parts = log.split('@')
            if len(parts) == 4:
                log_name, _, log_timestamp, log_action = parts
                if log_name == name:
                    try:
                        log_datetime = datetime.strptime(log_timestamp.split('.')[0], '%Y-%m-%d %H:%M:%S')
                        log_date = log_datetime.date()
                        
                        if last_date is None or log_datetime > last_date:
                            last_action = log_action
                            last_date = log_datetime
                    except:
                        continue
        
        if last_action is None:
            return True
        
        current_date = datetime.now().date()
        same_day = (last_date.date() == current_date) if last_date else False
        
        if not same_day:
            return True
        
        if last_action == 'Clock_In' and current_action == 'Clock_Out':
            return True
        elif last_action == 'Clock_Out' and current_action == 'Clock_In':
            return True
        elif last_action == current_action:
            return False
        
        return True
    
    def save_logs(self, name, role, action):
        """Save recognition logs to Redis after validation"""
        if name != 'Unknown':
            if self.check_last_action(name, action):
                current_time = str(datetime.now())
                concat_string = f"{name}@{role}@{current_time}@{action}"
                self.r.lpush('attendance:logs', concat_string)
                self.status_message = f"✔️ {action} recorded for {name}"
                self.status_message_time = time.time()
                return True
            else:
                self.status_message = f"❌ {name} already {action.replace('_', '-').lower()} today"
                self.status_message_time = time.time()
                return False
        return False
    
    def determine_action(self):
        """Determine whether to clock in or out based on current time"""
        current_hour = datetime.now().hour
        if 0 <= current_hour < 12:
            return 'Clock_In'
        else:
            return 'Clock_Out'
    
    def process_frame(self, frame):
        """Process each video frame for face detection and recognition"""
        # Determine current action based on time
        self.current_action = self.determine_action()
        
        # Perform face detection
        results = self.faceapp.get(frame)
        processed_frame = frame.copy()
        
        for res in results:
            x1, y1, x2, y2 = res['bbox'].astype(int)
            embeddings = res['embedding']
            person_name, person_role = self.ml_search_algorithm(embeddings)
            
            # Set box color based on recognition
            color = (0, 255, 0) if person_name != 'Unknown' else (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
            
            # Display name and role
            cv2.putText(processed_frame, person_name, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Check if it's time to take action
            current_time = time.time()
            if current_time - self.last_action_time >= self.action_interval:
                if person_name != 'Unknown':
                    self.save_logs(person_name, person_role, self.current_action)
                self.last_action_time = current_time
        
        # Display current action and status message
        cv2.putText(processed_frame, f"Current: {self.current_action.replace('_', '-')}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display status message if recent
        if time.time() - self.status_message_time < 5:
            cv2.putText(processed_frame, self.status_message, 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return processed_frame

def main():
    # Initialize the system
    system = FaceRecognitionSystem()
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device")
        return
    
    # Set window name with error handling
    window_name = "Face Recognition Attendance System"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        WINDOW_GUI = True
    except cv2.error:
        print("GUI not available - running in headless mode")
        WINDOW_GUI = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process the frame
            processed_frame = system.process_frame(frame)
            
            # Display the frame if GUI is available
            if WINDOW_GUI:
                cv2.imshow(window_name, processed_frame)
                if cv2.waitKey(1) == 27:  # ESC key
                    break
            else:
                # Add delay for headless mode
                time.sleep(0.1)
                
    finally:
        cap.release()
        if WINDOW_GUI:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()