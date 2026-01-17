import {create} from 'zustand';
import {LaneMetadata} from '../services/piClient';
import {UserLocation} from '../services/locationService';

interface NavigationState {
    currentLocation: UserLocation | null;
    piLaneData: LaneMetadata | null;
    recommendedLanes: number[];
    isConnectedToPi: boolean;
    setLocation: (location: UserLocation) => void;
    setPiLaneData: (data: LaneMetadata) => void;
    setRecommendedLanes: (lanes: number[]) => void;
    setPiConnectionStatus: (connected: boolean) => void;
}

export const useNavigationStore = create<NavigationState>((set) => ({
    currentLocation: null,
    piLaneData: null,
    recommendedLanes: [],
    isConnectedToPi: false,
    setLocation: (location) => set({ currentLocation: location}),
    setPiLaneData: (data) => set ({ piLaneData: data}),
    setRecommendedLanes: (lanes) => set ({ recommendedLanes: lanes}),
    setPiConnectionStatus: (connected) => set({ isConnectedToPi: connected}),
}));