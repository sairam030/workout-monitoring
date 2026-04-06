"""
Feature Engineering Module
==========================

Extract biomechanically meaningful features from raw landmarks

Why Feature Engineering?
- Raw coordinates are position/scale dependent
- Domain knowledge improves model performance
- Angles are invariant to camera position
- Reduces dimensionality while increasing signal

Feature Types:
1. Angular features (16 angles)
2. Distance features (10 distances)  
3. Velocity & acceleration (temporal)
4. Statistical features (sequence summary)
5. Movement smoothness metrics
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.distance import euclidean


class FeatureEngineer:
    """
    Extract domain-specific features from pose landmarks
    """
    
    # MediaPipe landmark indices
    LANDMARKS = {
        'nose': 0,
        'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13, 'right_elbow': 14,
        'left_wrist': 15, 'right_wrist': 16,
        'left_hip': 23, 'right_hip': 24,
        'left_knee': 25, 'right_knee': 26,
        'left_ankle': 27, 'right_ankle': 28,
        'left_heel': 29, 'right_heel': 30,
        'left_foot': 31, 'right_foot': 32
    }
    
    def __init__(self):
        """Initialize feature engineer"""
        pass
    
    def get_landmark_coords(self, landmarks, landmark_name):
        """
        Extract 3D coordinates for a specific landmark
        
        Args:
            landmarks: np.array shape (132,) - Flattened landmarks
            landmark_name: Name from LANDMARKS dict
            
        Returns:
            np.array: [x, y, z] coordinates
        """
        idx = self.LANDMARKS[landmark_name]
        # Each landmark has 4 values: x, y, z, visibility
        start = idx * 4
        return landmarks[start:start+3]  # Get x, y, z (skip visibility)
    
    def calculate_angle(self, p1, p2, p3):
        """
        Calculate angle at p2 formed by p1-p2-p3
        
        Mathematical Formula:
            cos(θ) = (v1 · v2) / (|v1| |v2|)
            θ = arccos(cos(θ))
            
        Why angles?
        - Invariant to position and scale
        - Directly measure joint flexion/extension
        - Biomechanically interpretable
        
        Args:
            p1, p2, p3: np.array [x, y, z] coordinates
            
        Returns:
            float: Angle in degrees [0, 180]
        """
        # Vectors from p2 to p1 and p2 to p3
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Dot product and magnitudes
        dot_product = np.dot(v1, v2)
        magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        # Avoid division by zero
        if magnitude == 0:
            return 0.0
        
        # Calculate angle
        cosine = dot_product / magnitude
        cosine = np.clip(cosine, -1.0, 1.0)  # Numerical stability
        angle = np.degrees(np.arccos(cosine))
        
        return angle
    
    def extract_angles(self, landmarks):
        """
        Extract all relevant joint angles (16 angles)
        
        Angles by Exercise:
        
        Squats:
        - Knee angles (L/R): Hip-Knee-Ankle
        - Hip angles (L/R): Shoulder-Hip-Knee
        - Back angle: Shoulder-Hip-vertical
        
        Push-ups:
        - Elbow angles (L/R): Shoulder-Elbow-Wrist
        - Shoulder angles (L/R): Hip-Shoulder-Elbow
        - Body line angle: Ankle-Hip-Shoulder
        
        Bicep Curls:
        - Elbow angles (L/R): Shoulder-Elbow-Wrist
        - Shoulder angles (L/R): Hip-Shoulder-Elbow
        - Forearm angles (L/R): Elbow-Wrist-Hand
        
        Returns:
            dict: {angle_name: value} - 16 angles
        """
        angles = {}
        
        # Left Knee Angle (Hip-Knee-Ankle)
        left_hip = self.get_landmark_coords(landmarks, 'left_hip')
        left_knee = self.get_landmark_coords(landmarks, 'left_knee')
        left_ankle = self.get_landmark_coords(landmarks, 'left_ankle')
        angles['left_knee'] = self.calculate_angle(left_hip, left_knee, left_ankle)
        
        # Right Knee Angle
        right_hip = self.get_landmark_coords(landmarks, 'right_hip')
        right_knee = self.get_landmark_coords(landmarks, 'right_knee')
        right_ankle = self.get_landmark_coords(landmarks, 'right_ankle')
        angles['right_knee'] = self.calculate_angle(right_hip, right_knee, right_ankle)
        
        # Left Hip Angle (Shoulder-Hip-Knee)
        left_shoulder = self.get_landmark_coords(landmarks, 'left_shoulder')
        angles['left_hip'] = self.calculate_angle(left_shoulder, left_hip, left_knee)
        
        # Right Hip Angle
        right_shoulder = self.get_landmark_coords(landmarks, 'right_shoulder')
        angles['right_hip'] = self.calculate_angle(right_shoulder, right_hip, right_knee)
        
        # Left Elbow Angle (Shoulder-Elbow-Wrist)
        left_elbow = self.get_landmark_coords(landmarks, 'left_elbow')
        left_wrist = self.get_landmark_coords(landmarks, 'left_wrist')
        angles['left_elbow'] = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        
        # Right Elbow Angle
        right_elbow = self.get_landmark_coords(landmarks, 'right_elbow')
        right_wrist = self.get_landmark_coords(landmarks, 'right_wrist')
        angles['right_elbow'] = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # Left Shoulder Angle (Hip-Shoulder-Elbow)
        angles['left_shoulder'] = self.calculate_angle(left_hip, left_shoulder, left_elbow)
        
        # Right Shoulder Angle
        angles['right_shoulder'] = self.calculate_angle(right_hip, right_shoulder, right_elbow)
        
        # Back Angle (vertical alignment)
        # Use average of left and right sides
        mid_hip = (left_hip + right_hip) / 2
        mid_shoulder = (left_shoulder + right_shoulder) / 2
        vertical_point = mid_hip + np.array([0, 1, 0])  # Point above hip
        angles['back_vertical'] = self.calculate_angle(mid_shoulder, mid_hip, vertical_point)
        
        # Body Line (Ankle-Hip-Shoulder) for push-ups
        mid_ankle = (left_ankle + right_ankle) / 2
        angles['body_line'] = self.calculate_angle(mid_ankle, mid_hip, mid_shoulder)
        
        return angles
    
    def extract_distances(self, landmarks):
        """
        Extract key inter-landmark distances (10 distances)
        
        Why distances?
        - Measure relative positioning (stance width, arm extension)
        - Normalized by body height for invariance
        - Detect form errors (e.g., knees caving in)
        
        Distances:
        1. Knee-to-knee (stance width)
        2. Ankle-to-ankle
        3. Shoulder-to-shoulder (body width)
        4. Wrist-to-wrist (arm span)
        5. Hip-to-ankle (body height reference)
        6. Elbow-to-hip
        7. Wrist-to-shoulder
        8. Hip-to-ground projection
        
        Returns:
            dict: {distance_name: normalized_value}
        """
        distances = {}
        
        # Get key landmarks
        left_shoulder = self.get_landmark_coords(landmarks, 'left_shoulder')
        right_shoulder = self.get_landmark_coords(landmarks, 'right_shoulder')
        left_elbow = self.get_landmark_coords(landmarks, 'left_elbow')
        right_elbow = self.get_landmark_coords(landmarks, 'right_elbow')
        left_wrist = self.get_landmark_coords(landmarks, 'left_wrist')
        right_wrist = self.get_landmark_coords(landmarks, 'right_wrist')
        left_hip = self.get_landmark_coords(landmarks, 'left_hip')
        right_hip = self.get_landmark_coords(landmarks, 'right_hip')
        left_knee = self.get_landmark_coords(landmarks, 'left_knee')
        right_knee = self.get_landmark_coords(landmarks, 'right_knee')
        left_ankle = self.get_landmark_coords(landmarks, 'left_ankle')
        right_ankle = self.get_landmark_coords(landmarks, 'right_ankle')
        
        # Body height reference (hip to ankle)
        mid_hip = (left_hip + right_hip) / 2
        mid_ankle = (left_ankle + right_ankle) / 2
        body_height = euclidean(mid_hip[:2], mid_ankle[:2])  # Use x,y only
        
        if body_height == 0:
            body_height = 1.0  # Avoid division by zero
        
        # Knee-to-knee (stance width)
        distances['knee_width'] = euclidean(left_knee[:2], right_knee[:2]) / body_height
        
        # Ankle-to-ankle
        distances['ankle_width'] = euclidean(left_ankle[:2], right_ankle[:2]) / body_height
        
        # Shoulder-to-shoulder
        distances['shoulder_width'] = euclidean(left_shoulder[:2], right_shoulder[:2]) / body_height
        
        # Wrist-to-wrist (arm span)
        distances['wrist_span'] = euclidean(left_wrist[:2], right_wrist[:2]) / body_height
        
        # Elbow-to-hip (left)
        distances['left_elbow_hip'] = euclidean(left_elbow[:2], left_hip[:2]) / body_height
        
        # Elbow-to-hip (right)
        distances['right_elbow_hip'] = euclidean(right_elbow[:2], right_hip[:2]) / body_height
        
        # Wrist-to-shoulder (left arm extension)
        distances['left_arm_extension'] = euclidean(left_wrist[:2], left_shoulder[:2]) / body_height
        
        # Wrist-to-shoulder (right arm extension)
        distances['right_arm_extension'] = euclidean(right_wrist[:2], right_shoulder[:2]) / body_height
        
        # Hip height (vertical position)
        distances['hip_height'] = mid_hip[1]  # y coordinate (normalized [0,1])
        
        # Body height reference (stored for completeness)
        distances['body_height_ref'] = body_height
        
        return distances
    
    def extract_velocities(self, landmarks_sequence):
        """
        Calculate velocities and accelerations from temporal sequence
        
        Why temporal features?
        - Capture movement dynamics
        - Identify phases (concentric/eccentric)
        - Detect jerky movements (poor form)
        
        Formulas:
            velocity[t] = (position[t] - position[t-1]) / dt
            acceleration[t] = (velocity[t] - velocity[t-1]) / dt
            jerk[t] = (acceleration[t] - acceleration[t-1]) / dt
        
        Args:
            landmarks_sequence: np.array shape (n_frames, 132)
            
        Returns:
            dict: {
                'velocities': np.array shape (n_frames-1, 132),
                'accelerations': np.array shape (n_frames-2, 132),
                'jerks': np.array shape (n_frames-3, 132)
            }
        """
        # Calculate velocities (first derivative)
        velocities = np.diff(landmarks_sequence, axis=0)
        
        # Calculate accelerations (second derivative)
        accelerations = np.diff(velocities, axis=0)
        
        # Calculate jerks (third derivative)
        jerks = np.diff(accelerations, axis=0)
        
        return {
            'velocities': velocities,
            'accelerations': accelerations,
            'jerks': jerks
        }
    
    def calculate_smoothness(self, landmarks_sequence):
        """
        Calculate movement smoothness using dimensionless jerk
        
        Why smoothness?
        - Correct form = smooth, controlled movement
        - Incorrect form = jerky, unstable movement
        
        Metric: Log Dimensionless Jerk (lower = smoother)
            LDLJ = -log(∫ jerk² dt)
        
        Args:
            landmarks_sequence: np.array shape (n_frames, 132)
            
        Returns:
            float: Smoothness score
        """
        # Calculate jerk
        motion = self.extract_velocities(landmarks_sequence)
        jerks = motion['jerks']
        
        # Mean squared jerk across all dimensions
        jerk_squared = np.mean(jerks ** 2)
        
        # Log dimensionless jerk (avoid log(0))
        ldlj = -np.log(jerk_squared + 1e-10)
        
        return ldlj
    
    def extract_sequence_statistics(self, landmarks_sequence):
        """
        Extract statistical features from entire sequence
        
        Why statistics?
        - Temporal summary of movement
        - Reduces variable-length sequences to fixed-size vectors
        - Captures overall movement characteristics
        
        Statistics per landmark:
        - Mean, Std, Min, Max
        - Range (Max - Min)
        - Percentiles (25th, 50th, 75th)
        
        Args:
            landmarks_sequence: np.array shape (n_frames, 132)
            
        Returns:
            dict: Statistical features
        """
        stats = {}
        
        # Per-landmark statistics
        stats['mean'] = np.mean(landmarks_sequence, axis=0)
        stats['std'] = np.std(landmarks_sequence, axis=0)
        stats['min'] = np.min(landmarks_sequence, axis=0)
        stats['max'] = np.max(landmarks_sequence, axis=0)
        stats['range'] = stats['max'] - stats['min']
        stats['median'] = np.median(landmarks_sequence, axis=0)
        stats['q25'] = np.percentile(landmarks_sequence, 25, axis=0)
        stats['q75'] = np.percentile(landmarks_sequence, 75, axis=0)
        
        # Temporal statistics
        stats['sequence_length'] = len(landmarks_sequence)
        stats['smoothness'] = self.calculate_smoothness(landmarks_sequence)
        
        return stats
    
    def engineer_features(self, landmarks_sequence):
        """
        Complete feature engineering pipeline
        
        Input: Raw landmarks sequence
        Output: Engineered feature vector
        
        Pipeline:
        1. Extract angles (frame-by-frame)
        2. Extract distances (frame-by-frame)
        3. Calculate velocities/accelerations
        4. Compute sequence statistics
        5. Aggregate into feature vector
        
        Args:
            landmarks_sequence: np.array shape (n_frames, 132)
            
        Returns:
            np.array: Engineered features (fixed size)
        """
        features = []
        
        # Frame-by-frame features
        angles_sequence = []
        distances_sequence = []
        
        for landmarks in landmarks_sequence:
            # Extract angles
            angles = self.extract_angles(landmarks)
            angles_sequence.append(list(angles.values()))
            
            # Extract distances
            distances = self.extract_distances(landmarks)
            distances_sequence.append(list(distances.values()))
        
        angles_sequence = np.array(angles_sequence)
        distances_sequence = np.array(distances_sequence)
        
        # Handle empty sequences (fallback)
        if len(angles_sequence) == 0:
            # Return zero features if no valid frames
            return np.zeros(72)
        
        # Statistics of angles
        angles_mean = np.mean(angles_sequence, axis=0)
        angles_std = np.std(angles_sequence, axis=0)
        angles_min = np.min(angles_sequence, axis=0)
        angles_max = np.max(angles_sequence, axis=0)
        angles_range = angles_max - angles_min  # Range of Motion (ROM)
        
        features.extend(angles_mean)
        features.extend(angles_std)
        features.extend(angles_range)
        
        # Statistics of distances
        distances_mean = np.mean(distances_sequence, axis=0)
        distances_std = np.std(distances_sequence, axis=0)
        
        features.extend(distances_mean)
        features.extend(distances_std)
        
        # Velocity statistics
        motion = self.extract_velocities(landmarks_sequence)
        vel_mean = np.mean(np.abs(motion['velocities']), axis=0)
        vel_max = np.max(np.abs(motion['velocities']), axis=0)
        
        features.extend(vel_mean[:10])  # Sample: first 10 velocity features
        features.extend(vel_max[:10])
        
        # Smoothness
        smoothness = self.calculate_smoothness(landmarks_sequence)
        features.append(smoothness)
        
        # Sequence length
        features.append(len(landmarks_sequence))
        
        return np.array(features)


if __name__ == "__main__":
    """
    Example usage and testing
    """
    print("Feature Engineering Module")
    print("=" * 60)
    
    # Create sample landmarks (30 frames, 132 features)
    np.random.seed(42)
    sample_sequence = np.random.rand(30, 132)
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Extract features
    print("\n1. Extracting angles from first frame...")
    angles = fe.extract_angles(sample_sequence[0])
    print(f"   Extracted {len(angles)} angles:")
    for name, value in list(angles.items())[:5]:
        print(f"   - {name}: {value:.1f}°")
    
    print("\n2. Extracting distances from first frame...")
    distances = fe.extract_distances(sample_sequence[0])
    print(f"   Extracted {len(distances)} distances:")
    for name, value in list(distances.items())[:5]:
        print(f"   - {name}: {value:.3f}")
    
    print("\n3. Computing temporal features...")
    motion = fe.extract_velocities(sample_sequence)
    print(f"   Velocities shape: {motion['velocities'].shape}")
    print(f"   Accelerations shape: {motion['accelerations'].shape}")
    print(f"   Jerks shape: {motion['jerks'].shape}")
    
    print("\n4. Calculating smoothness...")
    smoothness = fe.calculate_smoothness(sample_sequence)
    print(f"   Smoothness score: {smoothness:.3f}")
    
    print("\n5. Engineering complete feature vector...")
    features = fe.engineer_features(sample_sequence)
    print(f"   Final feature vector size: {len(features)}")
    print(f"   Features: {features[:10]}")  # Show first 10
    
    print("\n" + "=" * 60)
    print("✓ Feature engineering module ready!")
    print("\nFeature Counts:")
    print(f"  - Angles (mean, std, range): {len(angles) * 3}")
    print(f"  - Distances (mean, std): {len(distances) * 2}")
    print(f"  - Velocities (sample): 20")
    print(f"  - Smoothness: 1")
    print(f"  - Sequence length: 1")
    print(f"  Total: {len(features)} features")
