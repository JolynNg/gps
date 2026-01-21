# pi-lane-pipeline/src/lane_recommender.py
"""
Phase 5: Convert lane metrics + navigation to recommended_lanes.
"""
from typing import List, Optional, Dict, Any

def recommend_lanes(
    lane_count: int,
    current_lane_index: int,
    navigation_maneuver: Optional[str] = None,  # "left", "right", "straight", "exit_right"
    distance_to_turn: Optional[float] = None  # meters
) -> List[int]:
    """
    Recommend lanes based on navigation context.
    
    Args:
        lane_count: Total number of lanes
        current_lane_index: Current lane (0-indexed)
        navigation_maneuver: Next navigation instruction
        distance_to_turn: Distance to next turn in meters
        
    Returns:
        List of recommended lane indices (1-indexed for UI)
    """
    if lane_count == 0:
        return []
    
    # Default: stay in current lane
    recommended = [current_lane_index + 1]  # Convert to 1-indexed
    
    if navigation_maneuver is None:
        return recommended
    
    # Rule engine
    if "exit_right" in navigation_maneuver.lower() or "right" in navigation_maneuver.lower():
        # Need to be in right lanes
        if distance_to_turn and distance_to_turn < 200:  # Close to turn
            # Get rightmost lanes
            recommended = list(range(max(1, lane_count - 1), lane_count + 1))
        else:
            # Start moving right gradually
            target_lane = min(lane_count, current_lane_index + 2)
            recommended = [target_lane]
    
    elif "left" in navigation_maneuver.lower() and "exit" not in navigation_maneuver.lower():
        # Keep left or move left
        if distance_to_turn and distance_to_turn < 200:
            recommended = [1, 2]  # Leftmost lanes
        else:
            target_lane = max(1, current_lane_index)
            recommended = [target_lane]
    
    elif "straight" in navigation_maneuver.lower():
        # Stay in middle lanes if possible
        if lane_count >= 3:
            middle = lane_count // 2
            recommended = [middle, middle + 1]
        else:
            recommended = [current_lane_index + 1]
    
    # Ensure recommendations are valid
    recommended = [r for r in recommended if 1 <= r <= lane_count]
    
    return recommended if recommended else [current_lane_index + 1]