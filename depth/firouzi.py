#A time based cache cleaner could/should also be implemented for larger run sequances
import numpy as np
import cv2
from dv import AedatFile
import tonic
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
import time


class EventFrameStereoMatcher:
    def __init__(self, height: int, width: int, d_max: int, theta: float, epsilon: float,
                 neighborhood_size: int = 4, alpha: float = 0.3, beta: float = 0.002):
        self.height = height
        self.width = width
        self.d_max = d_max
        self.theta = theta
        self.epsilon = epsilon
        self.neighborhood_size = neighborhood_size 
        self.alpha = alpha
        self.beta = beta
        
        # Initialize cooperative cells for ON and OFF events
        self.cells_on = np.zeros((height, width, d_max + 1))
        self.cells_off = np.zeros((height, width, d_max + 1))
        self.cell_times_on = np.zeros((height, width, d_max + 1))
        self.cell_times_off = np.zeros((height, width, d_max + 1))

    def temporal_decay(self, time_diff: float) -> float:
        return 1 / (1 + self.beta * time_diff)

    def compute_excitatory_support(self, x: int, y: int, d: int, cells: np.ndarray, current_time: float) -> float:
        
        support = 0.0
        half_size = self.neighborhood_size // 2
        
        #for dy in range(-half_size, half_size + 1):
        #    for dx in range(-half_size, half_size + 1):
        #        ny, nx = y + dy, x + dx
        #        if 0 <= ny < self.height and 0 <= nx < self.width:
        #            weight = self.temporal_decay(time_diff)
        #            time_diff = current_time - self.cell_times_on[ny, nx, d]
         #           support += weight * cells[ny, nx, d]
                    
        # Debug: print excitatory support
        # print(f"[DEBUG] Excitatory support at ({x},{y},{d}): {support}")

        y_min = max(0, y - self.neighborhood_size // 2)
        y_max = min(self.height, y + self.neighborhood_size // 2 + 1)
        x_min = max(0, x - self.neighborhood_size // 2)
        x_max = min(self.width, x + self.neighborhood_size // 2 + 1)

        # Extract the neighborhood blocks for cells and cell times
        cell_block = cells[y_min:y_max, x_min:x_max, d]
        time_block = self.cell_times_on[y_min:y_max, x_min:x_max, d]

        # Calculate time difference for the entire block
        time_diff = current_time - time_block
        decay = 1 / (1 + self.beta * time_diff)
        support = np.sum(decay * cell_block)

        return support


    def compute_inhibitory_support(self, x: int, y: int, d: int, cells: np.ndarray, current_time: float) -> float:
        start_time_2 = time.time()
        support = 0.0
        for d_other in range(self.d_max + 1):
            if d_other != d:
                time_diff = current_time - self.cell_times_on[y, x, d_other]
                weight = self.temporal_decay(time_diff)
                support += weight * cells[y, x, d_other]
        
        # Debug: print inhibitory support
        # print(f"[DEBUG] Inhibitory support at ({x},{y},{d}): {support}")
      
        return support*0.7

    def compute_cell_activity(self, x: int, y: int, d: int, current_time: float,
                            left_frame: np.ndarray, right_frame: np.ndarray, cells: np.ndarray,
                            cell_times: np.ndarray) -> float:
        # Temporal decay based on time difference
        time_diff = current_time - cell_times[y, x, d]
        current_activity = cells[y, x, d] * self.temporal_decay(time_diff)
        
        # Compute excitatory support (for within-disparity continuity)
        excitatory = self.compute_excitatory_support(x, y, d, cells, current_time)
        
        # Compute inhibitory support (for cross-disparity uniqueness)
        inhibitory = self.compute_inhibitory_support(x, y, d, cells, current_time)
        
        # Matching cost between left and right frames (x - d)
        x_right = x - d
        # matching_cost = abs(int(left_frame[y, x]) - int(right_frame[y, x_right])) if x_right >= 0 else float('inf')
        # matching_weight = 1.0 / (1.0 + matching_cost) if matching_cost != float('inf') else 0.0


        matching_cost = abs(int(left_frame[y, x]) - int(right_frame[y, x_right])) + 1e-3 # Avoid high cost for small differences
        matching_weight = 1.0 / (1.0 + matching_cost)
        
        # Update cell activity considering both excitatory, inhibitory effects and matching cost
        new_activity = (current_activity + excitatory - (self.alpha * inhibitory) + matching_weight)
        
        return max(0.0, new_activity)


    def process_event_frames(self, left_frame: np.ndarray, right_frame: np.ndarray, 
                            current_time: float, polarity: str, cache=None):
        start_time_4 = time.time()  
        disparity_map = np.full((self.height, self.width), 0, dtype=np.float32)                  

        cells = self.cells_on if polarity == "ON" else self.cells_off
        cell_times = self.cell_times_on if polarity == "ON" else self.cell_times_off


        for y in range(self.height):
            for x in range(self.width):
                if left_frame[y, x] > 0:
                    activities = np.zeros(self.d_max + 1)
                    for d in range(self.d_max + 1):
                        cache_key = (x, y, d, polarity)#additional cache activity added, which reduces the run time in time
                        if cache_key in cache:
                            activity = cache[cache_key]
                        else:
                            activity = self.compute_cell_activity(
                            x, y, d, current_time, left_frame, right_frame, cells, cell_times)
                            cache[cache_key] = activity 

                        activities[d] = activity
                         # Cache the computed activity
                    # Apply Winner-Take-All to choose disparity with the highest activity
                    winner_d = np.argmax(activities)
                    winner_activity = activities[winner_d]

                    # Ensure disparity only if activity is above threshold
                    if winner_activity > self.theta:
                        disparity_map[y, x] = winner_d
                        cells[y, x, winner_d] = winner_activity
                        cell_times[y, x, winner_d] = current_time
                    else:
                        for d in range(self.d_max + 1):
                            cells[y, x, d] += self.epsilon
                        cell_times[y, x, :] = current_time
        
        
        #implemented for run time analysis
        end_time_4 = time.time()
        runtime_process_event_frames=  end_time_4 - start_time_4 
        print("process event  frames: ")
        print(runtime_process_event_frames)
        zero_count = np.size(disparity_map) - np.count_nonzero(disparity_map == 0)
        print("Number of 0 elements in array: ")
        print(zero_count)

        print(disparity_map.size)

        return disparity_map, cache


    def combine_on_off_disparities(self, disparity_map_on: np.ndarray, disparity_map_off: np.ndarray) -> np.ndarray:
        combined_map = np.maximum(disparity_map_on, disparity_map_off)
        return combined_map
    
    def plot_correspondence_points(self, left_frame: np.ndarray, right_frame: np.ndarray, disparity_map: np.ndarray):
        """Plot correspondence points between left and right frames using the disparity map."""
        
        plt.figure(figsize=(12, 6))

        
        # Display the left frame
        plt.subplot(1, 2, 1)
        plt.imshow(left_frame, cmap='gray')
        plt.title("Left Frame with Correspondence Points")
        
        # Display the right frame
        plt.subplot(1, 2, 2)
        plt.imshow(right_frame, cmap='gray')
        plt.title("Right Frame with Correspondence Points")
        
        # Plot points based on disparity map
        for y in range(self.height):
            for x in range(self.width):
                disparity = disparity_map[y, x]
                if disparity > 0:  # Ensure valid disparity
                    x_right = int(x - disparity)
                    if x_right >= 0:
                        # Plot the point in the left frame
                        plt.subplot(1, 2, 1)
                        plt.plot(x, y, 'ro', markersize=4)

                        # Plot the corresponding point in the right frame
                        plt.subplot(1, 2, 2)
                        plt.plot(x_right, y, 'go', markersize=4)

        
       
        plt.show()

def process_aedat_frames(aedat_path: str, stereo_map_path: str):
    # Initialize the EventFrameStereoMatcher with parameters from the paper

    
    cache = {}
    start_time_6 = time.time()  
    matcher = EventFrameStereoMatcher(
        height=260,
        width=346,
        d_max=108,
        theta=0.5,
        epsilon=0.1,
        neighborhood_size=4,
        alpha=0.5,
        beta=0.01
    )
    
    # Load AEDAT file events
    with AedatFile(aedat_path) as f:
        events_left = np.hstack([packet for packet in f['events_1'].numpy()])
        events_right = np.hstack([packet for packet in f['events'].numpy()])

    # Transform the event data into frames using tonic # width, height, 2 channels; time window accumulated to form single frame
    transform = tonic.transforms.ToFrame(sensor_size=(346, 260, 2), time_window=10000)
    dt = np.dtype([("x", "int"), ("y", "int"), ("t", "int"), ("p", "int")])
    data_left = np.stack((events_left['x'], events_left['y'], events_left['timestamp'], events_left['polarity'])).T
    data_right = np.stack((events_right['x'], events_right['y'], events_right['timestamp'], events_right['polarity'])).T
    data_left = rfn.unstructured_to_structured(data_left, dt)
    data_right = rfn.unstructured_to_structured(data_right, dt)
    frames_left = transform(data_left)
    frames_right = transform(data_right)
    
    # Split frames into ON and OFF polarities for left and right cameras
    frames_left_on = frames_left[:, 0, :, :].astype(np.uint8)*255
    frames_left_off = frames_left[:, 1, :, :].astype(np.uint8)*255
    frames_right_on = frames_right[:, 0, :, :].astype(np.uint8)*255
    frames_right_off = frames_right[:, 1, :, :].astype(np.uint8)*255
    
    # Load stereo rectification maps
    cv_file = cv2.FileStorage()
    cv_file.open(stereo_map_path, cv2.FileStorage_READ)
    stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
    stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
    stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
    stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
    cv_file.release()
    
    # Process each frame to generate disparity maps
    for i in range(185,200):
        # Rectify ON frames for left and right cameras
        rectified_left_on = cv2.remap(frames_left_on[i], stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        rectified_right_on = cv2.remap(frames_right_on[i], stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

        # Rectify OFF frames for left and right cameras
        rectified_left_off = cv2.remap(frames_left_off[i], stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        rectified_right_off = cv2.remap(frames_right_off[i], stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
       
        # Compute disparity maps for ON and OFF frames
        disparity_map_on, cache= matcher.process_event_frames(rectified_left_on, rectified_right_on, current_time=i, polarity="ON", cache = cache)
        disparity_map_off, _ = matcher.process_event_frames(rectified_left_off, rectified_right_off, current_time=i, polarity="OFF", cache = cache)
        matcher.plot_correspondence_points(rectified_left_on, rectified_right_on, disparity_map_on)

        # Combine ON and OFF disparity maps
        combined_disparity_map = matcher.combine_on_off_disparities(disparity_map_on, disparity_map_off)

        # Normalize the disparity map for color visualization
        normalized_disparity_map = cv2.normalize(combined_disparity_map, None, 0, 255, cv2.NORM_MINMAX)
        combined_disparity_map_8bit = np.uint8(normalized_disparity_map)
        #Appllied colormap
        colored_map = cv2.applyColorMap(combined_disparity_map_8bit, cv2.COLORMAP_JET)

        filename = f'pray_it_works_{i:04d}.png'
        cv2.imwrite(filename, colored_map)
        plt.show()

        # Debug: Print saved file name for each frame
        print(f"[DEBUG] Saved disparity map for frame {i} as {filename}")
        end_time_6 = time.time()
        runtime_process_aedat_frames= end_time_6 - start_time_6
        print("runtime_process_aedat_frames")
        print(runtime_process_aedat_frames)
# Example usage
if __name__ == "__main__":
    process_aedat_frames(aedat_path="2.aedat4", stereo_map_path="stereoMap1.xml")

